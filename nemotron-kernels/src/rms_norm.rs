use crate::KernelStub;
use crate::tensor::{GpuTensor, TensorError};

pub const SPEC: KernelStub = KernelStub {
    name: "rms_norm_host",
    summary: "RMSNorm and gated RMSNorm kernels.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmsNormBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RmsNormKernel {
    pub name: &'static str,
    pub backend: RmsNormBackend,
}

pub const RMS_NORM: RmsNormKernel = RmsNormKernel {
    name: "rms_norm_host",
    backend: RmsNormBackend::HostFallback,
};

pub const GATED_RMS_NORM: RmsNormKernel = RmsNormKernel {
    name: "gated_rms_norm_host",
    backend: RmsNormBackend::HostFallback,
};

pub fn supported_rms_norm_kernels() -> [RmsNormKernel; 2] {
    [RMS_NORM, GATED_RMS_NORM]
}

pub fn rms_norm_host(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    rms_norm_into_host(input, weight, epsilon, &mut output)?;
    Ok(output)
}

pub fn rms_norm_in_place_host(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<(), RmsNormError> {
    let output = rms_norm_host(values, weight, epsilon)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn rms_norm_into_host(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    validate_lengths(input, weight, None, output)?;
    apply_rms_norm(input, weight, None, epsilon, output)
}

pub fn gated_rms_norm_host(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    gated_rms_norm_into_host(input, weight, gate, epsilon, &mut output)?;
    Ok(output)
}

pub fn gated_rms_norm_in_place_host(
    values: &mut [f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<(), RmsNormError> {
    let output = gated_rms_norm_host(values, weight, gate, epsilon)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn gated_rms_norm_into_host(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    validate_lengths(input, weight, Some(gate), output)?;
    apply_rms_norm(input, weight, Some(gate), epsilon, output)
}

pub fn rms_host(input: &[f32], epsilon: f32) -> Result<f32, RmsNormError> {
    if input.is_empty() {
        return Err(RmsNormError::EmptyInput);
    }

    if epsilon < 0.0 {
        return Err(RmsNormError::NegativeEpsilon(epsilon));
    }

    let mean_square = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / input.len() as f64;
    Ok((mean_square as f32 + epsilon).sqrt())
}

fn apply_rms_norm(
    input: &[f32],
    weight: &[f32],
    gate: Option<&[f32]>,
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    let denom = rms_host(input, epsilon)?;
    let scale = denom.recip();

    for index in 0..input.len() {
        let mut value = input[index] * scale * weight[index];
        if let Some(gate) = gate {
            value *= gate[index];
        }
        output[index] = value;
    }

    Ok(())
}

fn validate_lengths(
    input: &[f32],
    weight: &[f32],
    gate: Option<&[f32]>,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    if input.is_empty() {
        return Err(RmsNormError::EmptyInput);
    }

    if input.len() != weight.len() {
        return Err(RmsNormError::LengthMismatch {
            expected: input.len(),
            actual: weight.len(),
            argument: "weight",
        });
    }

    if let Some(gate) = gate {
        if input.len() != gate.len() {
            return Err(RmsNormError::LengthMismatch {
                expected: input.len(),
                actual: gate.len(),
                argument: "gate",
            });
        }
    }

    if input.len() != output.len() {
        return Err(RmsNormError::LengthMismatch {
            expected: input.len(),
            actual: output.len(),
            argument: "output",
        });
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum RmsNormError {
    EmptyInput,
    NegativeEpsilon(f32),
    LengthMismatch {
        expected: usize,
        actual: usize,
        argument: &'static str,
    },
    DeviceError(String),
}


impl From<TensorError> for RmsNormError {
    fn from(e: TensorError) -> Self {
        RmsNormError::DeviceError(e.to_string())
    }
}


// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU RMS normalization.
pub async fn rms_norm(
    input: &GpuTensor,
    weight: &GpuTensor,
    epsilon: f32,
) -> Result<GpuTensor, RmsNormError> {
    let input_data = input.to_host_async().await?;
    let weight_data = weight.to_host_async().await?;
    let result = rms_norm_host(&input_data, &weight_data, epsilon)?;
    Ok(GpuTensor::from_host_async(&result, input.shape()).await?)
}

/// Async GPU gated RMS normalization.
pub async fn gated_rms_norm(
    input: &GpuTensor,
    weight: &GpuTensor,
    gate: &GpuTensor,
    epsilon: f32,
) -> Result<GpuTensor, RmsNormError> {
    let input_data = input.to_host_async().await?;
    let weight_data = weight.to_host_async().await?;
    let gate_data = gate.to_host_async().await?;
    let result = gated_rms_norm_host(&input_data, &weight_data, &gate_data, epsilon)?;
    Ok(GpuTensor::from_host_async(&result, input.shape()).await?)
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

    /// Verifies that both RMSNorm kernels report HostFallback as their backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backends_for_now() {
        assert_eq!(
            supported_rms_norm_kernels(),
            [
                RmsNormKernel {
                    name: "rms_norm_host",
                    backend: RmsNormBackend::HostFallback,
                },
                RmsNormKernel {
                    name: "gated_rms_norm_host",
                    backend: RmsNormBackend::HostFallback,
                },
            ]
        );
    }

    /// Verifies RMSNorm with unit weights against hand-computed reference values.
    ///
    /// This catches errors in the RMS denominator or normalization formula.
    #[test]
    fn rms_norm_matches_reference_values() {
        let output = rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], 1e-5).unwrap();

        approx_eq(output[0], 0.6324543);
        approx_eq(output[1], 1.2649086);
    }

    /// Verifies that per-element weights scale the normalized output.
    ///
    /// This catches missing or swapped weight multiplication.
    #[test]
    fn rms_norm_applies_weights() {
        let output = rms_norm_host(&[1.0, 2.0], &[2.0, 0.5], 1e-5).unwrap();

        approx_eq(output[0], 1.2649086);
        approx_eq(output[1], 0.6324543);
    }

    /// Verifies that gated RMSNorm multiplies the gate vector after normalization.
    ///
    /// This catches missing gate multiplication or wrong application order.
    #[test]
    fn gated_rms_norm_multiplies_gate_after_normalization() {
        let output = gated_rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], &[2.0, 0.5], 1e-5).unwrap();

        approx_eq(output[0], 1.2649086);
        approx_eq(output[1], 0.6324543);
    }

    /// Verifies that the in-place RMSNorm variant mutates the buffer correctly.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn rms_norm_in_place_updates_buffer() {
        let mut values = [1.0, 2.0];
        rms_norm_in_place_host(&mut values, &[1.0, 1.0], 1e-5).unwrap();

        approx_eq(values[0], 0.6324543);
        approx_eq(values[1], 1.2649086);
    }

    /// Verifies that the in-place gated RMSNorm variant mutates the buffer correctly.
    ///
    /// This catches issues where in-place writes are skipped or mis-ordered.
    #[test]
    fn gated_rms_norm_in_place_updates_buffer() {
        let mut values = [1.0, 2.0];
        gated_rms_norm_in_place_host(&mut values, &[1.0, 1.0], &[0.5, 2.0], 1e-5).unwrap();

        approx_eq(values[0], 0.31622714);
        approx_eq(values[1], 2.529817);
    }

    /// Verifies that all-zero input produces all-zero output (0/sqrt(eps) * w * 0 = 0).
    ///
    /// This catches NaN or infinity from dividing zero by a near-zero RMS.
    #[test]
    fn zero_input_stays_zero() {
        let output = rms_norm_host(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 1e-5).unwrap();
        assert_eq!(output, vec![0.0, 0.0, 0.0]);
    }

    /// Verifies that mismatched input/weight lengths are rejected.
    ///
    /// This catches missing weight length validation.
    #[test]
    fn rejects_length_mismatch() {
        let error = rms_norm_host(&[1.0, 2.0], &[1.0], 1e-5).unwrap_err();
        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 1,
                argument: "weight",
            }
        );
    }

    /// Verifies that a mismatched gate length is rejected.
    ///
    /// This catches missing gate length validation.
    #[test]
    fn rejects_gate_length_mismatch() {
        let error = gated_rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], &[1.0], 1e-5).unwrap_err();
        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 1,
                argument: "gate",
            }
        );
    }

    /// Verifies that a negative epsilon is rejected.
    ///
    /// This catches missing epsilon sign validation.
    #[test]
    fn rejects_negative_epsilon() {
        let error = rms_norm_host(&[1.0, 2.0], &[1.0, 1.0], -1.0).unwrap_err();
        assert_eq!(error, RmsNormError::NegativeEpsilon(-1.0));
    }

    /// Verifies that empty input is rejected.
    ///
    /// This catches division by zero in the mean-square computation.
    #[test]
    fn rejects_empty_input() {
        let error = rms_norm_host(&[], &[], 1e-5).unwrap_err();
        assert_eq!(error, RmsNormError::EmptyInput);
    }

    /// Verifies that the async GPU RMS norm matches the host fallback.
    /// This catches regressions in the GPU data transfer and computation.
    #[tokio::test]
    async fn gpu_rms_norm_matches_host_fallback() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.5, 1.0, 1.5, 2.0];
        let epsilon = 1e-5;
        let expected = rms_norm_host(&input, &weight, epsilon).unwrap();
        let gpu_input = GpuTensor::from_host(&input, &[4]).unwrap();
        let gpu_weight = GpuTensor::from_host(&weight, &[4]).unwrap();
        let result = super::rms_norm(&gpu_input, &gpu_weight, epsilon).await.unwrap();
        assert_eq!(result.to_host(), expected);
    }

}
