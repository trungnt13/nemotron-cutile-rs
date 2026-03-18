use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "softmax",
    summary: "Numerically stable softmax kernels.",
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
    name: "softmax",
    backend: SoftmaxBackend::HostFallback,
};

pub fn supported_softmax_kernels() -> [SoftmaxKernel; 1] {
    [SOFTMAX]
}

pub fn softmax(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    softmax_into(values, &mut output)
        .expect("softmax output buffer is allocated from input length and cannot mismatch");
    output
}

pub fn softmax_in_place(values: &mut [f32]) {
    let output = softmax(values);
    values.copy_from_slice(&output);
}

pub fn softmax_into(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SoftmaxError {
    LengthMismatch { input: usize, output: usize },
    ZeroPartition,
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

    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_softmax_kernels(),
            [SoftmaxKernel {
                name: "softmax",
                backend: SoftmaxBackend::HostFallback,
            }]
        );
    }

    #[test]
    fn softmax_matches_reference_values() {
        let output = softmax(&[1.0, 2.0, 3.0]);

        approx_eq(output[0], 0.09003057);
        approx_eq(output[1], 0.24472848);
        approx_eq(output[2], 0.66524094);
    }

    #[test]
    fn softmax_is_stable_for_large_inputs() {
        let output = softmax(&[1000.0, 1001.0, 1002.0]);

        approx_eq(output[0], 0.09003057);
        approx_eq(output[1], 0.24472848);
        approx_eq(output[2], 0.66524094);
    }

    #[test]
    fn softmax_outputs_sum_to_one() {
        let output = softmax(&[-10.0, 0.0, 10.0, 20.0]);
        let sum: f32 = output.iter().sum();

        approx_eq(sum, 1.0);
    }

    #[test]
    fn softmax_in_place_updates_buffer() {
        let mut values = [0.0, 0.0, 0.0];
        softmax_in_place(&mut values);

        approx_eq(values[0], 1.0 / 3.0);
        approx_eq(values[1], 1.0 / 3.0);
        approx_eq(values[2], 1.0 / 3.0);
    }

    #[test]
    fn softmax_into_rejects_length_mismatch() {
        let mut output = [0.0; 1];
        let error = softmax_into(&[1.0, 2.0], &mut output).unwrap_err();

        assert_eq!(
            error,
            SoftmaxError::LengthMismatch {
                input: 2,
                output: 1,
            }
        );
    }

    #[test]
    fn softmax_handles_empty_input() {
        assert_eq!(softmax(&[]), Vec::<f32>::new());
    }
}
