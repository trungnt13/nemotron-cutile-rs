use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "rms_norm",
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
    name: "rms_norm",
    backend: RmsNormBackend::HostFallback,
};

pub const GATED_RMS_NORM: RmsNormKernel = RmsNormKernel {
    name: "gated_rms_norm",
    backend: RmsNormBackend::HostFallback,
};

pub fn supported_rms_norm_kernels() -> [RmsNormKernel; 2] {
    [RMS_NORM, GATED_RMS_NORM]
}

pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    rms_norm_into(input, weight, epsilon, &mut output)?;
    Ok(output)
}

pub fn rms_norm_in_place(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<(), RmsNormError> {
    let output = rms_norm(values, weight, epsilon)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn rms_norm_into(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    validate_lengths(input, weight, None, output)?;
    apply_rms_norm(input, weight, None, epsilon, output)
}

pub fn gated_rms_norm(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    gated_rms_norm_into(input, weight, gate, epsilon, &mut output)?;
    Ok(output)
}

pub fn gated_rms_norm_in_place(
    values: &mut [f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
) -> Result<(), RmsNormError> {
    let output = gated_rms_norm(values, weight, gate, epsilon)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn gated_rms_norm_into(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    validate_lengths(input, weight, Some(gate), output)?;
    apply_rms_norm(input, weight, Some(gate), epsilon, output)
}

pub fn rms(input: &[f32], epsilon: f32) -> Result<f32, RmsNormError> {
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
    let denom = rms(input, epsilon)?;
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RmsNormError {
    EmptyInput,
    NegativeEpsilon(f32),
    LengthMismatch {
        expected: usize,
        actual: usize,
        argument: &'static str,
    },
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
    fn reports_host_fallback_backends_for_now() {
        assert_eq!(
            supported_rms_norm_kernels(),
            [
                RmsNormKernel {
                    name: "rms_norm",
                    backend: RmsNormBackend::HostFallback,
                },
                RmsNormKernel {
                    name: "gated_rms_norm",
                    backend: RmsNormBackend::HostFallback,
                },
            ]
        );
    }

    #[test]
    fn rms_norm_matches_reference_values() {
        let output = rms_norm(&[1.0, 2.0], &[1.0, 1.0], 1e-5).unwrap();

        approx_eq(output[0], 0.6324543);
        approx_eq(output[1], 1.2649086);
    }

    #[test]
    fn rms_norm_applies_weights() {
        let output = rms_norm(&[1.0, 2.0], &[2.0, 0.5], 1e-5).unwrap();

        approx_eq(output[0], 1.2649086);
        approx_eq(output[1], 0.6324543);
    }

    #[test]
    fn gated_rms_norm_multiplies_gate_after_normalization() {
        let output = gated_rms_norm(&[1.0, 2.0], &[1.0, 1.0], &[2.0, 0.5], 1e-5).unwrap();

        approx_eq(output[0], 1.2649086);
        approx_eq(output[1], 0.6324543);
    }

    #[test]
    fn rms_norm_in_place_updates_buffer() {
        let mut values = [1.0, 2.0];
        rms_norm_in_place(&mut values, &[1.0, 1.0], 1e-5).unwrap();

        approx_eq(values[0], 0.6324543);
        approx_eq(values[1], 1.2649086);
    }

    #[test]
    fn gated_rms_norm_in_place_updates_buffer() {
        let mut values = [1.0, 2.0];
        gated_rms_norm_in_place(&mut values, &[1.0, 1.0], &[0.5, 2.0], 1e-5).unwrap();

        approx_eq(values[0], 0.31622714);
        approx_eq(values[1], 2.529817);
    }

    #[test]
    fn zero_input_stays_zero() {
        let output = rms_norm(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 1e-5).unwrap();
        assert_eq!(output, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_length_mismatch() {
        let error = rms_norm(&[1.0, 2.0], &[1.0], 1e-5).unwrap_err();
        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 1,
                argument: "weight",
            }
        );
    }

    #[test]
    fn rejects_gate_length_mismatch() {
        let error = gated_rms_norm(&[1.0, 2.0], &[1.0, 1.0], &[1.0], 1e-5).unwrap_err();
        assert_eq!(
            error,
            RmsNormError::LengthMismatch {
                expected: 2,
                actual: 1,
                argument: "gate",
            }
        );
    }

    #[test]
    fn rejects_negative_epsilon() {
        let error = rms_norm(&[1.0, 2.0], &[1.0, 1.0], -1.0).unwrap_err();
        assert_eq!(error, RmsNormError::NegativeEpsilon(-1.0));
    }

    #[test]
    fn rejects_empty_input() {
        let error = rms_norm(&[], &[], 1e-5).unwrap_err();
        assert_eq!(error, RmsNormError::EmptyInput);
    }
}
