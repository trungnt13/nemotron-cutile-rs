use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "conv1d",
    summary: "Depthwise causal Conv1D kernels for Mamba-2.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Conv1dBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dKernel {
    pub name: &'static str,
    pub backend: Conv1dBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv1dShape {
    pub sequence_len: usize,
    pub channel_count: usize,
    pub kernel_size: usize,
}

impl Conv1dShape {
    pub const fn new(sequence_len: usize, channel_count: usize, kernel_size: usize) -> Self {
        Self {
            sequence_len,
            channel_count,
            kernel_size,
        }
    }

    pub const fn input_len(self) -> usize {
        self.sequence_len * self.channel_count
    }

    pub const fn weight_len(self) -> usize {
        self.channel_count * self.kernel_size
    }

    pub const fn output_len(self) -> usize {
        self.input_len()
    }
}

pub const DEPTHWISE_CAUSAL_CONV1D: Conv1dKernel = Conv1dKernel {
    name: "depthwise_causal_conv1d",
    backend: Conv1dBackend::HostFallback,
};

pub fn supported_conv1d_kernels() -> [Conv1dKernel; 1] {
    [DEPTHWISE_CAUSAL_CONV1D]
}

pub fn depthwise_causal_conv1d(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
) -> Result<Vec<f32>, Conv1dError> {
    let mut output = vec![0.0; shape.output_len()];
    depthwise_causal_conv1d_into(input, weights, shape, &mut output)?;
    Ok(output)
}

pub fn depthwise_causal_conv1d_into(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
    output: &mut [f32],
) -> Result<(), Conv1dError> {
    validate_shape(input, weights, shape, output)?;

    for timestep in 0..shape.sequence_len {
        for channel in 0..shape.channel_count {
            let mut acc = 0.0_f64;

            for tap in 0..shape.kernel_size {
                if timestep < tap {
                    break;
                }

                let input_timestep = timestep - tap;
                let input_index = input_timestep * shape.channel_count + channel;
                let weight_index = channel * shape.kernel_size + (shape.kernel_size - 1 - tap);

                acc += f64::from(input[input_index]) * f64::from(weights[weight_index]);
            }

            output[timestep * shape.channel_count + channel] = acc as f32;
        }
    }

    Ok(())
}

fn validate_shape(
    input: &[f32],
    weights: &[f32],
    shape: Conv1dShape,
    output: &mut [f32],
) -> Result<(), Conv1dError> {
    if shape.channel_count == 0 || shape.kernel_size == 0 {
        return Err(Conv1dError::InvalidShape(shape));
    }

    if input.len() != shape.input_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "input",
            expected: shape.input_len(),
            actual: input.len(),
        });
    }

    if weights.len() != shape.weight_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "weights",
            expected: shape.weight_len(),
            actual: weights.len(),
        });
    }

    if output.len() != shape.output_len() {
        return Err(Conv1dError::LengthMismatch {
            argument: "output",
            expected: shape.output_len(),
            actual: output.len(),
        });
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Conv1dError {
    InvalidShape(Conv1dShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths differ");
        for (index, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= 1e-6,
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}"
            );
        }
    }

    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_conv1d_kernels(),
            [Conv1dKernel {
                name: "depthwise_causal_conv1d",
                backend: Conv1dBackend::HostFallback,
            }]
        );
    }

    #[test]
    fn applies_single_channel_causal_filter() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weights = [2.0, 1.0];
        let output = depthwise_causal_conv1d(&input, &weights, Conv1dShape::new(4, 1, 2)).unwrap();

        approx_eq_slice(&output, &[1.0, 4.0, 7.0, 10.0]);
    }

    #[test]
    fn applies_depthwise_filter_per_channel() {
        let input = [
            1.0, 10.0, //
            2.0, 20.0, //
            3.0, 30.0, //
        ];
        let weights = [
            1.0, 0.5, //
            2.0, -1.0, //
        ];

        let output = depthwise_causal_conv1d(&input, &weights, Conv1dShape::new(3, 2, 2)).unwrap();

        approx_eq_slice(
            &output,
            &[
                0.5, -10.0, //
                2.0, 0.0, //
                3.5, 10.0, //
            ],
        );
    }

    #[test]
    fn kernel_size_one_is_per_channel_scaling() {
        let input = [
            1.0, -1.0, //
            2.0, -2.0, //
            3.0, -3.0, //
        ];
        let weights = [2.0, -0.5];

        let output = depthwise_causal_conv1d(&input, &weights, Conv1dShape::new(3, 2, 1)).unwrap();

        approx_eq_slice(
            &output,
            &[
                2.0, 0.5, //
                4.0, 1.0, //
                6.0, 1.5, //
            ],
        );
    }

    #[test]
    fn conv1d_into_writes_existing_buffer() {
        let input = [
            1.0, 10.0, //
            2.0, 20.0, //
        ];
        let weights = [
            1.0, 1.0, //
            1.0, 0.0, //
        ];
        let mut output = [-1.0; 4];

        depthwise_causal_conv1d_into(&input, &weights, Conv1dShape::new(2, 2, 2), &mut output)
            .unwrap();

        approx_eq_slice(&output, &[1.0, 0.0, 3.0, 10.0]);
    }

    #[test]
    fn empty_sequence_produces_empty_output() {
        let output = depthwise_causal_conv1d(&[], &[1.0, 2.0], Conv1dShape::new(0, 1, 2)).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn rejects_invalid_shape() {
        let error = depthwise_causal_conv1d(&[], &[], Conv1dShape::new(1, 0, 1)).unwrap_err();
        assert_eq!(error, Conv1dError::InvalidShape(Conv1dShape::new(1, 0, 1)));
    }

    #[test]
    fn rejects_input_length_mismatch() {
        let error =
            depthwise_causal_conv1d(&[1.0], &[1.0, 2.0], Conv1dShape::new(2, 1, 2)).unwrap_err();

        assert_eq!(
            error,
            Conv1dError::LengthMismatch {
                argument: "input",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_weight_length_mismatch() {
        let error =
            depthwise_causal_conv1d(&[1.0, 2.0], &[1.0], Conv1dShape::new(2, 1, 2)).unwrap_err();

        assert_eq!(
            error,
            Conv1dError::LengthMismatch {
                argument: "weights",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_output_length_mismatch() {
        let mut output = [0.0; 1];
        let error = depthwise_causal_conv1d_into(
            &[1.0, 2.0],
            &[1.0, 0.5],
            Conv1dShape::new(2, 1, 2),
            &mut output,
        )
        .unwrap_err();

        assert_eq!(
            error,
            Conv1dError::LengthMismatch {
                argument: "output",
                expected: 2,
                actual: 1,
            }
        );
    }
}
