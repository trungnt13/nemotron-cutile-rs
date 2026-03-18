use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "ssm",
    summary: "Selective scan kernels for chunked Mamba-2 state updates.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SsmBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SsmKernel {
    pub name: &'static str,
    pub backend: SsmBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectiveScanShape {
    pub sequence_len: usize,
    pub channel_count: usize,
    pub state_size: usize,
}

impl SelectiveScanShape {
    pub const fn new(sequence_len: usize, channel_count: usize, state_size: usize) -> Self {
        Self {
            sequence_len,
            channel_count,
            state_size,
        }
    }

    pub const fn input_len(self) -> usize {
        self.sequence_len * self.channel_count
    }

    pub const fn state_matrix_len(self) -> usize {
        self.channel_count * self.state_size
    }

    pub const fn initial_state_len(self) -> usize {
        self.state_matrix_len()
    }

    pub const fn dt_len(self) -> usize {
        self.input_len()
    }

    pub const fn b_len(self) -> usize {
        self.sequence_len * self.channel_count * self.state_size
    }

    pub const fn c_len(self) -> usize {
        self.b_len()
    }

    pub const fn d_len(self) -> usize {
        self.channel_count
    }

    pub const fn output_len(self) -> usize {
        self.input_len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectiveScanParams<'a> {
    pub input: &'a [f32],
    pub delta_t: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub c: &'a [f32],
    pub d: Option<&'a [f32]>,
    pub initial_state: Option<&'a [f32]>,
    pub delta_bias: f32,
    pub apply_softplus_to_dt: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectiveScanOutput {
    pub values: Vec<f32>,
    pub final_state: Vec<f32>,
}

pub const SELECTIVE_SCAN: SsmKernel = SsmKernel {
    name: "selective_scan",
    backend: SsmBackend::HostFallback,
};

pub fn supported_ssm_kernels() -> [SsmKernel; 1] {
    [SELECTIVE_SCAN]
}

pub fn selective_scan(
    params: SelectiveScanParams<'_>,
    shape: SelectiveScanShape,
) -> Result<SelectiveScanOutput, SsmError> {
    let mut values = vec![0.0; shape.output_len()];
    let mut final_state = vec![0.0; shape.initial_state_len()];
    selective_scan_into(params, shape, &mut values, &mut final_state)?;
    Ok(SelectiveScanOutput {
        values,
        final_state,
    })
}

pub fn selective_scan_into(
    params: SelectiveScanParams<'_>,
    shape: SelectiveScanShape,
    output: &mut [f32],
    final_state: &mut [f32],
) -> Result<(), SsmError> {
    validate_params(&params, shape, output, final_state)?;

    if let Some(initial_state) = params.initial_state {
        final_state.copy_from_slice(initial_state);
    } else {
        final_state.fill(0.0);
    }

    for timestep in 0..shape.sequence_len {
        for channel in 0..shape.channel_count {
            let input_index = timestep * shape.channel_count + channel;
            let input_value = params.input[input_index];
            let raw_dt = params.delta_t[input_index] + params.delta_bias;
            let dt = if params.apply_softplus_to_dt {
                softplus(raw_dt)
            } else {
                raw_dt
            };

            if !dt.is_finite() || dt < 0.0 {
                return Err(SsmError::InvalidDeltaT {
                    timestep,
                    channel,
                    value: dt,
                });
            }

            let state_offset = channel * shape.state_size;
            let mut output_acc = 0.0_f64;

            for state_index in 0..shape.state_size {
                let matrix_index = state_offset + state_index;
                let transition = (dt * params.a[matrix_index]).exp();
                let input_mix = dt
                    * params.b[ssm_tensor_offset(timestep, channel, state_index, shape)]
                    * input_value;

                let next_state = transition * final_state[matrix_index] + input_mix;
                final_state[matrix_index] = next_state;

                let projection =
                    params.c[ssm_tensor_offset(timestep, channel, state_index, shape)] * next_state;
                output_acc += f64::from(projection);
            }

            if let Some(direct) = params.d {
                output_acc += f64::from(direct[channel] * input_value);
            }

            output[input_index] = output_acc as f32;
        }
    }

    Ok(())
}

fn validate_params(
    params: &SelectiveScanParams<'_>,
    shape: SelectiveScanShape,
    output: &mut [f32],
    final_state: &mut [f32],
) -> Result<(), SsmError> {
    if shape.sequence_len == 0 || shape.channel_count == 0 || shape.state_size == 0 {
        return Err(SsmError::InvalidShape(shape));
    }

    validate_len("input", params.input.len(), shape.input_len())?;
    validate_len("delta_t", params.delta_t.len(), shape.dt_len())?;
    validate_len("a", params.a.len(), shape.state_matrix_len())?;
    validate_len("b", params.b.len(), shape.b_len())?;
    validate_len("c", params.c.len(), shape.c_len())?;

    if let Some(direct) = params.d {
        validate_len("d", direct.len(), shape.d_len())?;
    }

    if let Some(initial_state) = params.initial_state {
        validate_len(
            "initial_state",
            initial_state.len(),
            shape.initial_state_len(),
        )?;
    }

    validate_len("output", output.len(), shape.output_len())?;
    validate_len("final_state", final_state.len(), shape.initial_state_len())?;

    Ok(())
}

fn validate_len(argument: &'static str, actual: usize, expected: usize) -> Result<(), SsmError> {
    if actual != expected {
        return Err(SsmError::LengthMismatch {
            argument,
            expected,
            actual,
        });
    }

    Ok(())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn ssm_tensor_offset(
    timestep: usize,
    channel: usize,
    state_index: usize,
    shape: SelectiveScanShape,
) -> usize {
    ((timestep * shape.channel_count + channel) * shape.state_size) + state_index
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SsmError {
    InvalidShape(SelectiveScanShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidDeltaT {
        timestep: usize,
        channel: usize,
        value: f32,
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
                diff <= 1e-5,
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}"
            );
        }
    }

    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_ssm_kernels(),
            [SsmKernel {
                name: "selective_scan",
                backend: SsmBackend::HostFallback,
            }]
        );
    }

    #[test]
    fn selective_scan_updates_state_and_output() {
        let shape = SelectiveScanShape::new(2, 1, 2);
        let output = selective_scan(
            SelectiveScanParams {
                input: &[1.0, 2.0],
                delta_t: &[1.0, 1.0],
                a: &[0.0, 0.0],
                b: &[
                    1.0, 0.5, //
                    1.0, 0.5, //
                ],
                c: &[
                    0.25, 1.0, //
                    0.25, 1.0, //
                ],
                d: Some(&[0.1]),
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();

        approx_eq_slice(&output.values, &[0.85, 2.45]);
        approx_eq_slice(&output.final_state, &[3.0, 1.5]);
    }

    #[test]
    fn selective_scan_uses_initial_state() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let output = selective_scan(
            SelectiveScanParams {
                input: &[2.0],
                delta_t: &[1.0],
                a: &[0.0],
                b: &[0.5],
                c: &[2.0],
                d: None,
                initial_state: Some(&[3.0]),
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap();

        approx_eq_slice(&output.values, &[8.0]);
        approx_eq_slice(&output.final_state, &[4.0]);
    }

    #[test]
    fn selective_scan_applies_softplus_to_delta_t() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let output = selective_scan(
            SelectiveScanParams {
                input: &[2.0],
                delta_t: &[-1.0],
                a: &[0.0],
                b: &[1.0],
                c: &[1.0],
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: true,
            },
            shape,
        )
        .unwrap();

        approx_eq_slice(&output.values, &[0.6265234]);
        approx_eq_slice(&output.final_state, &[0.6265234]);
    }

    #[test]
    fn selective_scan_into_writes_existing_buffers() {
        let shape = SelectiveScanShape::new(2, 1, 1);
        let params = SelectiveScanParams {
            input: &[1.0, 1.0],
            delta_t: &[1.0, 1.0],
            a: &[0.0],
            b: &[1.0, 1.0],
            c: &[1.0, 1.0],
            d: None,
            initial_state: None,
            delta_bias: 0.0,
            apply_softplus_to_dt: false,
        };
        let mut output = [-1.0; 2];
        let mut final_state = [-1.0; 1];

        selective_scan_into(params, shape, &mut output, &mut final_state).unwrap();

        approx_eq_slice(&output, &[1.0, 2.0]);
        approx_eq_slice(&final_state, &[2.0]);
    }

    #[test]
    fn rejects_invalid_shape() {
        let shape = SelectiveScanShape::new(0, 1, 1);
        let error = selective_scan(
            SelectiveScanParams {
                input: &[],
                delta_t: &[],
                a: &[1.0],
                b: &[],
                c: &[],
                d: Some(&[1.0]),
                initial_state: Some(&[0.0]),
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap_err();

        assert_eq!(error, SsmError::InvalidShape(shape));
    }

    #[test]
    fn rejects_length_mismatch() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let error = selective_scan(
            SelectiveScanParams {
                input: &[1.0],
                delta_t: &[1.0],
                a: &[1.0],
                b: &[],
                c: &[1.0],
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SsmError::LengthMismatch {
                argument: "b",
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn rejects_negative_delta_t_without_softplus() {
        let shape = SelectiveScanShape::new(1, 1, 1);
        let error = selective_scan(
            SelectiveScanParams {
                input: &[1.0],
                delta_t: &[-0.5],
                a: &[0.0],
                b: &[1.0],
                c: &[1.0],
                d: None,
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: false,
            },
            shape,
        )
        .unwrap_err();

        assert_eq!(
            error,
            SsmError::InvalidDeltaT {
                timestep: 0,
                channel: 0,
                value: -0.5,
            }
        );
    }
}
