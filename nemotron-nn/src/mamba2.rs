use crate::{LayerStub, LinearError, LinearProjection};
use nemotron_kernels::activations::silu_in_place_host;
use nemotron_kernels::conv1d::{depthwise_causal_conv1d_host, Conv1dError, Conv1dShape};
use nemotron_kernels::rms_norm::{gated_rms_norm_host, RmsNormError};
use nemotron_kernels::ssm::{
    selective_scan_host, SelectiveScanParams, SelectiveScanShape, SsmError,
};
use nemotron_kernels::tensor::GpuTensor;
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "mamba2",
    summary: "Host-fallback Mamba-2 mixer built from conv, scan, and gated RMSNorm stages.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mamba2ForwardShape {
    pub batch_size: usize,
    pub sequence_len: usize,
}

impl Mamba2ForwardShape {
    pub const fn new(batch_size: usize, sequence_len: usize) -> Self {
        Self {
            batch_size,
            sequence_len,
        }
    }

    pub const fn row_count(self) -> usize {
        self.batch_size * self.sequence_len
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2Cache {
    batch_size: usize,
    conv_channels: usize,
    conv_kernel_size: usize,
    state_size: usize,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl Mamba2Cache {
    pub fn new_zeroed(
        batch_size: usize,
        conv_channels: usize,
        conv_kernel_size: usize,
        state_size: usize,
    ) -> Self {
        let conv_state_len = batch_size * conv_channels * conv_kernel_size.saturating_sub(1);
        let ssm_state_len = batch_size * conv_channels * state_size;
        Self {
            batch_size,
            conv_channels,
            conv_kernel_size,
            state_size,
            conv_state: vec![0.0; conv_state_len],
            ssm_state: vec![0.0; ssm_state_len],
        }
    }

    pub fn conv_state(&self) -> &[f32] {
        &self.conv_state
    }

    pub fn ssm_state(&self) -> &[f32] {
        &self.ssm_state
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mamba2Mixer {
    hidden_size: usize,
    conv_channels: usize,
    state_size: usize,
    conv_kernel_size: usize,
    epsilon: f32,
    input_projection: LinearProjection,
    delta_t_projection: LinearProjection,
    b_projection: LinearProjection,
    c_projection: LinearProjection,
    gate_projection: LinearProjection,
    output_projection: LinearProjection,
    conv_weights: Vec<f32>,
    ssm_a: Vec<f32>,
    direct_term: Option<Vec<f32>>,
    rms_norm_weight: Vec<f32>,
}

impl Mamba2Mixer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        conv_channels: usize,
        state_size: usize,
        conv_kernel_size: usize,
        epsilon: f32,
        input_projection: LinearProjection,
        delta_t_projection: LinearProjection,
        b_projection: LinearProjection,
        c_projection: LinearProjection,
        gate_projection: LinearProjection,
        output_projection: LinearProjection,
        conv_weights: Vec<f32>,
        ssm_a: Vec<f32>,
        direct_term: Option<Vec<f32>>,
        rms_norm_weight: Vec<f32>,
    ) -> Result<Self, Mamba2Error> {
        validate_mamba_dims(
            hidden_size,
            conv_channels,
            state_size,
            conv_kernel_size,
            epsilon,
        )?;
        validate_projection(
            "input_projection",
            &input_projection,
            hidden_size,
            conv_channels,
        )?;
        validate_projection(
            "delta_t_projection",
            &delta_t_projection,
            hidden_size,
            conv_channels,
        )?;
        validate_projection(
            "b_projection",
            &b_projection,
            hidden_size,
            conv_channels * state_size,
        )?;
        validate_projection(
            "c_projection",
            &c_projection,
            hidden_size,
            conv_channels * state_size,
        )?;
        validate_projection(
            "gate_projection",
            &gate_projection,
            hidden_size,
            conv_channels,
        )?;
        validate_projection(
            "output_projection",
            &output_projection,
            conv_channels,
            hidden_size,
        )?;

        if conv_weights.len() != conv_channels * conv_kernel_size {
            return Err(Mamba2Error::LengthMismatch {
                argument: "conv_weights",
                expected: conv_channels * conv_kernel_size,
                actual: conv_weights.len(),
            });
        }

        if ssm_a.len() != conv_channels * state_size {
            return Err(Mamba2Error::LengthMismatch {
                argument: "ssm_a",
                expected: conv_channels * state_size,
                actual: ssm_a.len(),
            });
        }

        if let Some(direct_term) = direct_term.as_ref() {
            if direct_term.len() != conv_channels {
                return Err(Mamba2Error::LengthMismatch {
                    argument: "direct_term",
                    expected: conv_channels,
                    actual: direct_term.len(),
                });
            }
        }

        if rms_norm_weight.len() != conv_channels {
            return Err(Mamba2Error::LengthMismatch {
                argument: "rms_norm_weight",
                expected: conv_channels,
                actual: rms_norm_weight.len(),
            });
        }

        Ok(Self {
            hidden_size,
            conv_channels,
            state_size,
            conv_kernel_size,
            epsilon,
            input_projection,
            delta_t_projection,
            b_projection,
            c_projection,
            gate_projection,
            output_projection,
            conv_weights,
            ssm_a,
            direct_term,
            rms_norm_weight,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        shape: Mamba2ForwardShape,
        mut cache: Option<&mut Mamba2Cache>,
    ) -> Result<Vec<f32>, Mamba2Error> {
        validate_forward_shape(shape)?;
        validate_hidden_states(hidden_states, shape.row_count(), self.hidden_size)?;

        if let Some(cache) = cache.as_ref() {
            validate_cache(
                cache,
                shape.batch_size,
                self.conv_channels,
                self.conv_kernel_size,
                self.state_size,
            )?;
        }

        let row_count = shape.row_count();
        let projected_input = self
            .input_projection
            .project(hidden_states, row_count)
            .map_err(|source| Mamba2Error::Projection {
                projection: "input_projection",
                source,
            })?;
        let delta_t = self
            .delta_t_projection
            .project(hidden_states, row_count)
            .map_err(|source| Mamba2Error::Projection {
                projection: "delta_t_projection",
                source,
            })?;
        let b = self
            .b_projection
            .project(hidden_states, row_count)
            .map_err(|source| Mamba2Error::Projection {
                projection: "b_projection",
                source,
            })?;
        let c = self
            .c_projection
            .project(hidden_states, row_count)
            .map_err(|source| Mamba2Error::Projection {
                projection: "c_projection",
                source,
            })?;
        let gate = self
            .gate_projection
            .project(hidden_states, row_count)
            .map_err(|source| Mamba2Error::Projection {
                projection: "gate_projection",
                source,
            })?;

        let mut output = vec![0.0; row_count * self.hidden_size];
        let conv_prefix_steps = self.conv_kernel_size.saturating_sub(1);

        for batch in 0..shape.batch_size {
            let batch_input = batch_matrix_slice(
                &projected_input,
                batch,
                shape.sequence_len,
                self.conv_channels,
            );
            let conv_input = if conv_prefix_steps == 0 {
                batch_input.to_vec()
            } else {
                let mut values = Vec::with_capacity(
                    (conv_prefix_steps + shape.sequence_len) * self.conv_channels,
                );
                if let Some(cache) = cache.as_ref() {
                    values.extend_from_slice(batch_conv_state_slice(
                        cache,
                        batch,
                        self.conv_channels,
                        conv_prefix_steps,
                    ));
                } else {
                    values.resize(conv_prefix_steps * self.conv_channels, 0.0);
                }
                values.extend_from_slice(batch_input);
                values
            };

            let conv_output = depthwise_causal_conv1d_host(
                &conv_input,
                &self.conv_weights,
                Conv1dShape::new(
                    conv_input.len() / self.conv_channels,
                    self.conv_channels,
                    self.conv_kernel_size,
                ),
            )
            .map_err(Mamba2Error::Conv1d)?;
            let mut conv_activated =
                conv_output[(conv_prefix_steps * self.conv_channels)..].to_vec();
            silu_in_place_host(&mut conv_activated);

            let batch_delta_t =
                batch_matrix_slice(&delta_t, batch, shape.sequence_len, self.conv_channels);
            let batch_b = batch_matrix_slice(
                &b,
                batch,
                shape.sequence_len,
                self.conv_channels * self.state_size,
            );
            let batch_c = batch_matrix_slice(
                &c,
                batch,
                shape.sequence_len,
                self.conv_channels * self.state_size,
            );
            let batch_gate =
                batch_matrix_slice(&gate, batch, shape.sequence_len, self.conv_channels);

            let initial_state = cache.as_ref().map(|cache| {
                batch_ssm_state_slice(cache, batch, self.conv_channels, self.state_size).to_vec()
            });
            let scan_output = selective_scan_host(
                SelectiveScanParams {
                    input: &conv_activated,
                    delta_t: batch_delta_t,
                    a: &self.ssm_a,
                    b: batch_b,
                    c: batch_c,
                    d: self.direct_term.as_deref(),
                    initial_state: initial_state.as_deref(),
                    delta_bias: 0.0,
                    apply_softplus_to_dt: true,
                },
                SelectiveScanShape::new(shape.sequence_len, self.conv_channels, self.state_size),
            )
            .map_err(Mamba2Error::Ssm)?;

            let normalized = gated_rms_norm_rows(
                &scan_output.values,
                &self.rms_norm_weight,
                batch_gate,
                self.conv_channels,
                self.epsilon,
            )
            .map_err(Mamba2Error::RmsNorm)?;
            let projected_output = self
                .output_projection
                .project(&normalized, shape.sequence_len)
                .map_err(|source| Mamba2Error::Projection {
                    projection: "output_projection",
                    source,
                })?;

            let output_batch =
                batch_matrix_slice_mut(&mut output, batch, shape.sequence_len, self.hidden_size);
            output_batch.copy_from_slice(&projected_output);

            if let Some(cache) = cache.as_deref_mut() {
                if conv_prefix_steps > 0 {
                    let updated_state =
                        &conv_input[conv_input.len() - conv_prefix_steps * self.conv_channels..];
                    batch_conv_state_slice_mut(cache, batch, self.conv_channels, conv_prefix_steps)
                        .copy_from_slice(updated_state);
                }
                batch_ssm_state_slice_mut(cache, batch, self.conv_channels, self.state_size)
                    .copy_from_slice(&scan_output.final_state);
            }
        }

        Ok(output)
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn conv_channels(&self) -> usize {
        self.conv_channels
    }

    pub const fn state_size(&self) -> usize {
        self.state_size
    }

    pub const fn conv_kernel_size(&self) -> usize {
        self.conv_kernel_size
    }

    /// Async GPU Mamba2 forward. Delegates to host fallback via data transfer.
    pub async fn forward_gpu(
        &self,
        hidden_states: &GpuTensor,
        shape: Mamba2ForwardShape,
        cache: Option<&mut Mamba2Cache>,
    ) -> Result<GpuTensor, Mamba2Error> {
        let data = hidden_states
            .to_host_async()
            .await
            .map_err(|e| Mamba2Error::DeviceError(e.to_string()))?;
        let result = self.forward(&data, shape, cache)?;
        GpuTensor::from_host_async(&result, &[shape.row_count(), self.hidden_size])
            .await
            .map_err(|e| Mamba2Error::DeviceError(e.to_string()))
    }
}

fn validate_mamba_dims(
    hidden_size: usize,
    conv_channels: usize,
    state_size: usize,
    conv_kernel_size: usize,
    epsilon: f32,
) -> Result<(), Mamba2Error> {
    if hidden_size == 0 || conv_channels == 0 || state_size == 0 || conv_kernel_size == 0 {
        return Err(Mamba2Error::InvalidConfig {
            hidden_size,
            conv_channels,
            state_size,
            conv_kernel_size,
        });
    }

    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(Mamba2Error::InvalidEpsilon(epsilon));
    }

    Ok(())
}

fn validate_projection(
    name: &'static str,
    projection: &LinearProjection,
    input_dim: usize,
    output_dim: usize,
) -> Result<(), Mamba2Error> {
    if projection.input_dim() != input_dim || projection.output_dim() != output_dim {
        return Err(Mamba2Error::ProjectionShapeMismatch {
            projection: name,
            expected_input_dim: input_dim,
            actual_input_dim: projection.input_dim(),
            expected_output_dim: output_dim,
            actual_output_dim: projection.output_dim(),
        });
    }

    Ok(())
}

fn validate_forward_shape(shape: Mamba2ForwardShape) -> Result<(), Mamba2Error> {
    if shape.batch_size == 0 || shape.sequence_len == 0 {
        return Err(Mamba2Error::InvalidForwardShape(shape));
    }

    Ok(())
}

fn validate_hidden_states(
    hidden_states: &[f32],
    row_count: usize,
    hidden_size: usize,
) -> Result<(), Mamba2Error> {
    let expected = row_count * hidden_size;
    if hidden_states.len() != expected {
        return Err(Mamba2Error::LengthMismatch {
            argument: "hidden_states",
            expected,
            actual: hidden_states.len(),
        });
    }

    Ok(())
}

fn validate_cache(
    cache: &Mamba2Cache,
    batch_size: usize,
    conv_channels: usize,
    conv_kernel_size: usize,
    state_size: usize,
) -> Result<(), Mamba2Error> {
    let expected = Mamba2Cache::new_zeroed(batch_size, conv_channels, conv_kernel_size, state_size);
    if cache.batch_size != batch_size
        || cache.conv_channels != conv_channels
        || cache.conv_kernel_size != conv_kernel_size
        || cache.state_size != state_size
        || cache.conv_state.len() != expected.conv_state.len()
        || cache.ssm_state.len() != expected.ssm_state.len()
    {
        return Err(Mamba2Error::CacheShapeMismatch {
            batch_size,
            conv_channels,
            conv_kernel_size,
            state_size,
        });
    }

    Ok(())
}

fn batch_matrix_slice<'a>(
    values: &'a [f32],
    batch: usize,
    row_count: usize,
    width: usize,
) -> &'a [f32] {
    let start = batch * row_count * width;
    &values[start..start + row_count * width]
}

fn batch_matrix_slice_mut<'a>(
    values: &'a mut [f32],
    batch: usize,
    row_count: usize,
    width: usize,
) -> &'a mut [f32] {
    let start = batch * row_count * width;
    &mut values[start..start + row_count * width]
}

fn batch_conv_state_slice<'a>(
    cache: &'a Mamba2Cache,
    batch: usize,
    conv_channels: usize,
    conv_prefix_steps: usize,
) -> &'a [f32] {
    let start = batch * conv_channels * conv_prefix_steps;
    &cache.conv_state[start..start + conv_channels * conv_prefix_steps]
}

fn batch_conv_state_slice_mut<'a>(
    cache: &'a mut Mamba2Cache,
    batch: usize,
    conv_channels: usize,
    conv_prefix_steps: usize,
) -> &'a mut [f32] {
    let start = batch * conv_channels * conv_prefix_steps;
    &mut cache.conv_state[start..start + conv_channels * conv_prefix_steps]
}

fn batch_ssm_state_slice<'a>(
    cache: &'a Mamba2Cache,
    batch: usize,
    conv_channels: usize,
    state_size: usize,
) -> &'a [f32] {
    let start = batch * conv_channels * state_size;
    &cache.ssm_state[start..start + conv_channels * state_size]
}

fn batch_ssm_state_slice_mut<'a>(
    cache: &'a mut Mamba2Cache,
    batch: usize,
    conv_channels: usize,
    state_size: usize,
) -> &'a mut [f32] {
    let start = batch * conv_channels * state_size;
    &mut cache.ssm_state[start..start + conv_channels * state_size]
}

fn gated_rms_norm_rows(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    row_width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let row_count = input.len() / row_width;
    let mut output = vec![0.0; input.len()];

    for row_index in 0..row_count {
        let start = row_index * row_width;
        let end = start + row_width;
        let normalized =
            gated_rms_norm_host(&input[start..end], weight, &gate[start..end], epsilon)?;
        output[start..end].copy_from_slice(&normalized);
    }

    Ok(output)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mamba2Error {
    InvalidConfig {
        hidden_size: usize,
        conv_channels: usize,
        state_size: usize,
        conv_kernel_size: usize,
    },
    InvalidEpsilon(f32),
    InvalidForwardShape(Mamba2ForwardShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    ProjectionShapeMismatch {
        projection: &'static str,
        expected_input_dim: usize,
        actual_input_dim: usize,
        expected_output_dim: usize,
        actual_output_dim: usize,
    },
    CacheShapeMismatch {
        batch_size: usize,
        conv_channels: usize,
        conv_kernel_size: usize,
        state_size: usize,
    },
    Projection {
        projection: &'static str,
        source: LinearError,
    },
    Conv1d(Conv1dError),
    Ssm(SsmError),
    RmsNorm(RmsNormError),
    DeviceError(String),
}

impl fmt::Display for Mamba2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig {
                hidden_size,
                conv_channels,
                state_size,
                conv_kernel_size,
            } => write!(
                f,
                "mamba2 dimensions must be non-zero, got hidden_size={hidden_size}, conv_channels={conv_channels}, state_size={state_size}, conv_kernel_size={conv_kernel_size}"
            ),
            Self::InvalidEpsilon(epsilon) => write!(f, "epsilon must be finite and non-negative, got {epsilon}"),
            Self::InvalidForwardShape(shape) => write!(
                f,
                "mamba2 forward shape must be non-zero, got batch_size={}, sequence_len={}",
                shape.batch_size, shape.sequence_len
            ),
            Self::LengthMismatch {
                argument,
                expected,
                actual,
            } => write!(
                f,
                "{argument} length mismatch: expected {expected}, got {actual}"
            ),
            Self::ProjectionShapeMismatch {
                projection,
                expected_input_dim,
                actual_input_dim,
                expected_output_dim,
                actual_output_dim,
            } => write!(
                f,
                "{projection} shape mismatch: expected ({expected_input_dim} -> {expected_output_dim}), got ({actual_input_dim} -> {actual_output_dim})"
            ),
            Self::CacheShapeMismatch {
                batch_size,
                conv_channels,
                conv_kernel_size,
                state_size,
            } => write!(
                f,
                "cache shape mismatch for batch_size={batch_size}, conv_channels={conv_channels}, conv_kernel_size={conv_kernel_size}, state_size={state_size}"
            ),
            Self::Projection { projection, source } => write!(f, "{projection} failed: {source}"),
            Self::Conv1d(source) => write!(f, "conv1d failed: {source:?}"),
            Self::Ssm(source) => write!(f, "ssm failed: {source:?}"),
            Self::RmsNorm(source) => write!(f, "rms norm failed: {source:?}"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
        }
    }
}

impl Error for Mamba2Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use nemotron_kernels::activations::silu_host;
    use nemotron_kernels::ssm::{selective_scan_host, SelectiveScanParams, SelectiveScanShape};

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

    fn dense_projection(
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> LinearProjection {
        LinearProjection::new_dense_f32(input_dim, output_dim, weights, bias).unwrap()
    }

    fn scalar_identity_projection() -> LinearProjection {
        dense_projection(1, 1, vec![1.0], None)
    }

    fn scalar_bias_projection(bias: f32) -> LinearProjection {
        dense_projection(1, 1, vec![0.0], Some(vec![bias]))
    }

    fn simple_mixer(conv_kernel_size: usize) -> Mamba2Mixer {
        Mamba2Mixer::new(
            1,
            1,
            1,
            conv_kernel_size,
            1e-5,
            scalar_identity_projection(),
            scalar_bias_projection(1.0),
            scalar_bias_projection(1.0),
            scalar_bias_projection(1.0),
            scalar_bias_projection(1.0),
            scalar_identity_projection(),
            vec![1.0; conv_kernel_size],
            vec![0.0],
            None,
            vec![1.0],
        )
        .unwrap()
    }

    /// Verifies that the full Mamba2 forward (project → conv1d → SiLU → SSM → gated-RMSNorm → out-project)
    /// matches a manual step-by-step pipeline with kernel_size=1 (no conv prefix).
    ///
    /// This catches incorrect kernel composition ordering or shape wiring.
    #[test]
    fn forward_matches_manual_pipeline_without_cache() {
        let mixer = simple_mixer(1);
        let hidden_states = [1.0, 2.0];
        let shape = Mamba2ForwardShape::new(1, 2);

        let projected = mixer.input_projection.project(&hidden_states, 2).unwrap();
        let conv = silu_host(
            &depthwise_causal_conv1d_host(
                &projected,
                &mixer.conv_weights,
                Conv1dShape::new(2, 1, 1),
            )
            .unwrap(),
        );
        let dt = mixer.delta_t_projection.project(&hidden_states, 2).unwrap();
        let b = mixer.b_projection.project(&hidden_states, 2).unwrap();
        let c = mixer.c_projection.project(&hidden_states, 2).unwrap();
        let gate = mixer.gate_projection.project(&hidden_states, 2).unwrap();
        let scan = selective_scan_host(
            SelectiveScanParams {
                input: &conv,
                delta_t: &dt,
                a: &mixer.ssm_a,
                b: &b,
                c: &c,
                d: mixer.direct_term.as_deref(),
                initial_state: None,
                delta_bias: 0.0,
                apply_softplus_to_dt: true,
            },
            SelectiveScanShape::new(2, 1, 1),
        )
        .unwrap();
        let normalized = gated_rms_norm_rows(
            &scan.values,
            &mixer.rms_norm_weight,
            &gate,
            1,
            mixer.epsilon,
        )
        .unwrap();
        let expected = mixer.output_projection.project(&normalized, 2).unwrap();

        let actual = mixer.forward(&hidden_states, shape, None).unwrap();
        approx_eq_slice(&actual, &expected);
    }

    /// Verifies that incremental (cached) decode produces the same last-token output as a full-sequence forward.
    ///
    /// This catches conv-state or SSM-state update bugs that break autoregressive consistency.
    #[test]
    fn cache_keeps_decode_consistent_with_full_sequence() {
        let mixer = simple_mixer(2);
        let mut cache = Mamba2Cache::new_zeroed(1, 1, 2, 1);

        let first = mixer
            .forward(&[1.0, 2.0], Mamba2ForwardShape::new(1, 2), Some(&mut cache))
            .unwrap();
        assert_eq!(first.len(), 2);

        let next = mixer
            .forward(&[3.0], Mamba2ForwardShape::new(1, 1), Some(&mut cache))
            .unwrap();
        let full = mixer
            .forward(&[1.0, 2.0, 3.0], Mamba2ForwardShape::new(1, 3), None)
            .unwrap();

        approx_eq_slice(&next, &full[2..3]);
        approx_eq_slice(cache.conv_state(), &[3.0]);
        assert_eq!(cache.ssm_state().len(), 1);
    }

    /// Verifies that construction rejects a projection whose output dim doesn't match conv_channels.
    ///
    /// This catches weakened projection-shape validation in `Mamba2Mixer::new`.
    #[test]
    fn rejects_projection_shape_mismatch() {
        let error = Mamba2Mixer::new(
            1,
            1,
            1,
            1,
            1e-5,
            dense_projection(1, 2, vec![1.0, 0.0], None),
            scalar_bias_projection(1.0),
            scalar_bias_projection(1.0),
            scalar_bias_projection(1.0),
            scalar_bias_projection(1.0),
            scalar_identity_projection(),
            vec![1.0],
            vec![0.0],
            None,
            vec![1.0],
        )
        .unwrap_err();

        assert_eq!(
            error,
            Mamba2Error::ProjectionShapeMismatch {
                projection: "input_projection",
                expected_input_dim: 1,
                actual_input_dim: 1,
                expected_output_dim: 1,
                actual_output_dim: 2,
            }
        );
    }

    /// Verifies that forward rejects a cache whose batch_size doesn't match the forward shape.
    ///
    /// This catches missing or incorrect cache-shape validation.
    #[test]
    fn rejects_cache_shape_mismatch() {
        let mixer = simple_mixer(2);
        let mut cache = Mamba2Cache::new_zeroed(2, 1, 2, 1);

        let error = mixer
            .forward(&[1.0, 2.0], Mamba2ForwardShape::new(1, 2), Some(&mut cache))
            .unwrap_err();

        assert_eq!(
            error,
            Mamba2Error::CacheShapeMismatch {
                batch_size: 1,
                conv_channels: 1,
                conv_kernel_size: 2,
                state_size: 1,
            }
        );
    }

    /// Verifies that the GPU Mamba2 forward path matches the host-fallback output.
    ///
    /// This catches regressions in the async GPU data transfer path for Mamba2 mixer.
    #[tokio::test]
    async fn gpu_mamba2_forward_matches_host() {
        use nemotron_kernels::tensor::GpuTensor;
        let mixer = simple_mixer(2);
        let hidden = [1.0_f32, 0.5];
        let shape = Mamba2ForwardShape::new(1, 2);

        let host_result = mixer.forward(&hidden, shape, None).unwrap();

        let gpu_input = GpuTensor::from_host(&hidden, &[2, 1]).unwrap();
        let gpu_result = mixer.forward_gpu(&gpu_input, shape, None).await.unwrap();
        let gpu_host = gpu_result.to_host();

        approx_eq_slice(&gpu_host, &host_result);
    }
}
