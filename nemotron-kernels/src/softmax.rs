use crate::KernelStub;
use crate::tensor::{GpuTensor, TensorError};

pub const SPEC: KernelStub = KernelStub {
    name: "softmax_host",
    summary: "Numerically stable softmax_host kernels.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoftmaxBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoftmaxKernel {
    pub name: &'static str,
    pub backend: SoftmaxBackend,
}

pub const SOFTMAX: SoftmaxKernel = SoftmaxKernel {
    name: "softmax_host",
    backend: SoftmaxBackend::HostFallback,
};

pub fn supported_softmax_kernels() -> [SoftmaxKernel; 1] {
    [SOFTMAX]
}

pub fn softmax_host(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    softmax_into_host(values, &mut output)
        .expect("softmax_host output buffer is allocated from input length and cannot mismatch");
    output
}

pub fn softmax_in_place_host(values: &mut [f32]) {
    let output = softmax_host(values);
    values.copy_from_slice(&output);
}

pub fn softmax_into_host(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
    if input.len() != output.len() {
        return Err(SoftmaxError::LengthMismatch {
            input: input.len(),
            output: output.len(),
        });
    }

    if input.is_empty() {
        return Ok(());
    }

    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;

    for (value, slot) in input.iter().copied().zip(output.iter_mut()) {
        let weight = (value - max).exp();
        *slot = weight;
        sum += f64::from(weight);
    }

    if sum == 0.0 {
        return Err(SoftmaxError::ZeroPartition);
    }

    let inv_sum = (1.0 / sum) as f32;
    for value in output.iter_mut() {
        *value *= inv_sum;
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SoftmaxError {
    LengthMismatch { input: usize, output: usize },
    ZeroPartition,
    DeviceError(String),
}


impl From<TensorError> for SoftmaxError {
    fn from(e: TensorError) -> Self {
        SoftmaxError::DeviceError(e.to_string())
    }
}


// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU softmax normalization.
pub async fn softmax(input: &GpuTensor) -> Result<GpuTensor, SoftmaxError> {
    let data = input.to_host_async().await.map_err(|e| SoftmaxError::DeviceError(e.to_string()))?;
    let result = softmax_host(&data);
    GpuTensor::from_host_async(&result, input.shape())
        .await
        .map_err(|e| SoftmaxError::DeviceError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(lhs: f32, rhs: f32) {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1e-6,
            "values differ: left={lhs:?}, right={rhs:?}, diff={diff:?}"
        );
    }

    /// Verifies that the softmax_host kernel reports HostFallback as its backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_softmax_kernels(),
            [SoftmaxKernel {
                name: "softmax_host",
                backend: SoftmaxBackend::HostFallback,
            }]
        );
    }

    /// Verifies softmax_host output against known reference values for [1, 2, 3].
    ///
    /// This catches errors in the exp/normalize formula.
    #[test]
    fn softmax_matches_reference_values() {
        let output = softmax_host(&[1.0, 2.0, 3.0]);

        approx_eq(output[0], 0.09003057);
        approx_eq(output[1], 0.24472848);
        approx_eq(output[2], 0.66524094);
    }

    /// Verifies that large inputs produce the same result as shifted inputs (numerical stability).
    ///
    /// This catches missing max-subtraction in the softmax_host, which would cause overflow.
    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let output = softmax_host(&[1000.0, 1001.0, 1002.0]);

        approx_eq(output[0], 0.09003057);
        approx_eq(output[1], 0.24472848);
        approx_eq(output[2], 0.66524094);
    }

    /// Verifies that softmax_host outputs sum to 1.0 across a wide value range.
    ///
    /// This catches partition normalization errors.
    #[test]
    fn softmax_outputs_sum_to_one() {
        let output = softmax_host(&[-10.0, 0.0, 10.0, 20.0]);
        let sum: f32 = output.iter().sum();

        approx_eq(sum, 1.0);
    }

    /// Verifies that the in-place variant produces uniform probabilities for equal inputs.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn softmax_in_place_updates_buffer() {
        let mut values = [0.0, 0.0, 0.0];
        softmax_in_place_host(&mut values);

        approx_eq(values[0], 1.0 / 3.0);
        approx_eq(values[1], 1.0 / 3.0);
        approx_eq(values[2], 1.0 / 3.0);
    }

    /// Verifies that mismatched input/output lengths are rejected.
    ///
    /// This catches missing length validation in the _into variant.
    #[test]
    fn softmax_into_rejects_length_mismatch() {
        let mut output = [0.0; 1];
        let error = softmax_into_host(&[1.0, 2.0], &mut output).unwrap_err();

        assert_eq!(
            error,
            SoftmaxError::LengthMismatch {
                input: 2,
                output: 1,
            }
        );
    }

    /// Verifies that empty input produces empty output without error.
    ///
    /// This catches panics on zero-length slices.
    #[test]
    fn softmax_handles_empty_input() {
        assert_eq!(softmax_host(&[]), Vec::<f32>::new());
    }

    /// Verifies that the async GPU softmax matches the host fallback.
    /// This catches regressions in the GPU data transfer path.
    #[tokio::test]
    async fn gpu_softmax_matches_host_fallback() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let expected = softmax_host(&data);
        let gpu_input = GpuTensor::from_host(&data, &[4]).unwrap();
        let result = super::softmax(&gpu_input).await.unwrap();
        assert_eq!(result.to_host(), expected);
    }

}
