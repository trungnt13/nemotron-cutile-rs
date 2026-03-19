use crate::{LayerStub, LinearError, LinearProjection};
use nemotron_kernels::attention::{
    scaled_dot_product_attention_host, AttentionError, AttentionOptions, AttentionShape,
    GROUPED_QUERY_ATTENTION,
};
use nemotron_kernels::tensor::GpuTensor;
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "attention",
    summary: "GQA attention layer wrapping host attention kernels.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionForwardShape {
    pub batch_size: usize,
    pub query_sequence_len: usize,
    pub key_value_sequence_len: usize,
}

impl AttentionForwardShape {
    pub const fn new(
        batch_size: usize,
        query_sequence_len: usize,
        key_value_sequence_len: usize,
    ) -> Self {
        Self {
            batch_size,
            query_sequence_len,
            key_value_sequence_len,
        }
    }

    pub const fn query_row_count(self) -> usize {
        self.batch_size * self.query_sequence_len
    }

    pub const fn key_value_row_count(self) -> usize {
        self.batch_size * self.key_value_sequence_len
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionLayer {
    hidden_size: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_dim: usize,
    query_projection: LinearProjection,
    key_projection: LinearProjection,
    value_projection: LinearProjection,
    output_projection: LinearProjection,
}

impl AttentionLayer {
    pub fn new(
        hidden_size: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_dim: usize,
        query_projection: LinearProjection,
        key_projection: LinearProjection,
        value_projection: LinearProjection,
        output_projection: LinearProjection,
    ) -> Result<Self, AttentionLayerError> {
        validate_head_shape(
            hidden_size,
            query_head_count,
            key_value_head_count,
            head_dim,
        )?;

        let query_projection_dim = query_head_count * head_dim;
        let key_value_projection_dim = key_value_head_count * head_dim;

        validate_projection(
            "query_projection",
            &query_projection,
            hidden_size,
            query_projection_dim,
        )?;
        validate_projection(
            "key_projection",
            &key_projection,
            hidden_size,
            key_value_projection_dim,
        )?;
        validate_projection(
            "value_projection",
            &value_projection,
            hidden_size,
            key_value_projection_dim,
        )?;
        validate_projection(
            "output_projection",
            &output_projection,
            query_projection_dim,
            hidden_size,
        )?;

        Ok(Self {
            hidden_size,
            query_head_count,
            key_value_head_count,
            head_dim,
            query_projection,
            key_projection,
            value_projection,
            output_projection,
        })
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn query_head_count(&self) -> usize {
        self.query_head_count
    }

    pub const fn key_value_head_count(&self) -> usize {
        self.key_value_head_count
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn kernel(&self) -> nemotron_kernels::attention::AttentionKernel {
        GROUPED_QUERY_ATTENTION
    }

    pub fn forward_self_attention(
        &self,
        hidden_states: &[f32],
        batch_size: usize,
        sequence_len: usize,
        options: AttentionOptions,
    ) -> Result<Vec<f32>, AttentionLayerError> {
        self.forward(
            hidden_states,
            hidden_states,
            AttentionForwardShape::new(batch_size, sequence_len, sequence_len),
            options,
        )
    }

    pub fn forward(
        &self,
        query_states: &[f32],
        key_value_states: &[f32],
        shape: AttentionForwardShape,
        options: AttentionOptions,
    ) -> Result<Vec<f32>, AttentionLayerError> {
        validate_forward_shape(shape)?;
        validate_hidden_states(
            "query_states",
            query_states,
            shape.query_row_count(),
            self.hidden_size,
        )?;
        validate_hidden_states(
            "key_value_states",
            key_value_states,
            shape.key_value_row_count(),
            self.hidden_size,
        )?;

        let query = self
            .query_projection
            .project(query_states, shape.query_row_count())
            .map_err(|source| AttentionLayerError::Projection {
                projection: "query_projection",
                source,
            })?;
        let key = self
            .key_projection
            .project(key_value_states, shape.key_value_row_count())
            .map_err(|source| AttentionLayerError::Projection {
                projection: "key_projection",
                source,
            })?;
        let value = self
            .value_projection
            .project(key_value_states, shape.key_value_row_count())
            .map_err(|source| AttentionLayerError::Projection {
                projection: "value_projection",
                source,
            })?;

        let attention_shape = AttentionShape::new(
            shape.batch_size,
            shape.query_sequence_len,
            shape.key_value_sequence_len,
            self.query_head_count,
            self.key_value_head_count,
            self.head_dim,
        );
        let attention_output =
            scaled_dot_product_attention_host(&query, &key, &value, attention_shape, options)
                .map_err(AttentionLayerError::Attention)?;

        self.output_projection
            .project(&attention_output, shape.query_row_count())
            .map_err(|source| AttentionLayerError::Projection {
                projection: "output_projection",
                source,
            })
    }

    /// Async GPU self-attention. Delegates to host fallback via data transfer.
    pub async fn forward_self_attention_gpu(
        &self,
        hidden_states: &GpuTensor,
        batch_size: usize,
        sequence_len: usize,
        options: AttentionOptions,
    ) -> Result<GpuTensor, AttentionLayerError> {
        self.forward_gpu(
            hidden_states,
            hidden_states,
            AttentionForwardShape::new(batch_size, sequence_len, sequence_len),
            options,
        )
        .await
    }

    /// Async GPU attention forward. Delegates to host fallback via data transfer.
    pub async fn forward_gpu(
        &self,
        query_states: &GpuTensor,
        key_value_states: &GpuTensor,
        shape: AttentionForwardShape,
        options: AttentionOptions,
    ) -> Result<GpuTensor, AttentionLayerError> {
        let q_host = query_states.to_host_async().await.map_err(|e| AttentionLayerError::DeviceError(e.to_string()))?;
        let kv_host = key_value_states.to_host_async().await.map_err(|e| AttentionLayerError::DeviceError(e.to_string()))?;
        let result = self.forward(&q_host, &kv_host, shape, options)?;
        let out_rows = shape.query_row_count();
        GpuTensor::from_host_async(&result, &[out_rows, self.hidden_size])
            .await
            .map_err(|e| AttentionLayerError::DeviceError(e.to_string()))
    }
}

fn validate_head_shape(
    hidden_size: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_dim: usize,
) -> Result<(), AttentionLayerError> {
    if hidden_size == 0 || query_head_count == 0 || key_value_head_count == 0 || head_dim == 0 {
        return Err(AttentionLayerError::InvalidHeadShape {
            hidden_size,
            query_head_count,
            key_value_head_count,
            head_dim,
        });
    }

    if !query_head_count.is_multiple_of(key_value_head_count) {
        return Err(AttentionLayerError::InvalidHeadGrouping {
            query_head_count,
            key_value_head_count,
        });
    }

    Ok(())
}

fn validate_projection(
    name: &'static str,
    projection: &LinearProjection,
    input_dim: usize,
    output_dim: usize,
) -> Result<(), AttentionLayerError> {
    if projection.input_dim() != input_dim || projection.output_dim() != output_dim {
        return Err(AttentionLayerError::ProjectionShapeMismatch {
            projection: name,
            expected_input_dim: input_dim,
            actual_input_dim: projection.input_dim(),
            expected_output_dim: output_dim,
            actual_output_dim: projection.output_dim(),
        });
    }

    Ok(())
}

fn validate_forward_shape(shape: AttentionForwardShape) -> Result<(), AttentionLayerError> {
    if shape.batch_size == 0 || shape.query_sequence_len == 0 || shape.key_value_sequence_len == 0 {
        return Err(AttentionLayerError::InvalidForwardShape(shape));
    }

    Ok(())
}

fn validate_hidden_states(
    argument: &'static str,
    hidden_states: &[f32],
    row_count: usize,
    hidden_size: usize,
) -> Result<(), AttentionLayerError> {
    let expected = row_count * hidden_size;
    if hidden_states.len() != expected {
        return Err(AttentionLayerError::LengthMismatch {
            argument,
            expected,
            actual: hidden_states.len(),
        });
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttentionLayerError {
    InvalidHeadShape {
        hidden_size: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_dim: usize,
    },
    InvalidHeadGrouping {
        query_head_count: usize,
        key_value_head_count: usize,
    },
    InvalidForwardShape(AttentionForwardShape),
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
    Projection {
        projection: &'static str,
        source: LinearError,
    },
    Attention(AttentionError),
    DeviceError(String),
}

impl fmt::Display for AttentionLayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeadShape {
                hidden_size,
                query_head_count,
                key_value_head_count,
                head_dim,
            } => write!(
                f,
                "attention dimensions must be non-zero, got hidden_size={hidden_size}, query_head_count={query_head_count}, key_value_head_count={key_value_head_count}, head_dim={head_dim}"
            ),
            Self::InvalidHeadGrouping {
                query_head_count,
                key_value_head_count,
            } => write!(
                f,
                "query_head_count ({query_head_count}) must be divisible by key_value_head_count ({key_value_head_count})"
            ),
            Self::InvalidForwardShape(shape) => write!(
                f,
                "attention forward shape must be non-zero, got batch_size={}, query_sequence_len={}, key_value_sequence_len={}",
                shape.batch_size, shape.query_sequence_len, shape.key_value_sequence_len
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
            Self::Projection { projection, source } => {
                write!(f, "{projection} failed: {source}")
            }
            Self::Attention(source) => write!(f, "attention kernel failed: {source:?}"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
        }
    }
}

impl Error for AttentionLayerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use nemotron_kernels::attention::scaled_dot_product_attention_host;

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

    fn identity_projection(size: usize) -> LinearProjection {
        let mut weights = vec![0.0; size * size];
        for index in 0..size {
            weights[index * size + index] = 1.0;
        }
        LinearProjection::new_dense_f32(size, size, weights, None).unwrap()
    }

    /// Verifies that self-attention with identity projections reproduces the raw kernel output.
    ///
    /// This catches incorrect projection wiring or shape mismatches between layer and kernel.
    #[test]
    fn self_attention_matches_kernel_when_projections_are_identity() {
        let layer = AttentionLayer::new(
            2,
            1,
            1,
            2,
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
        )
        .unwrap();
        let hidden_states = [1.0, 0.0, 0.0, 1.0];
        let options = AttentionOptions {
            softmax_scale: Some(1.0),
            ..AttentionOptions::default()
        };

        let expected = scaled_dot_product_attention_host(
            &hidden_states,
            &hidden_states,
            &hidden_states,
            AttentionShape::new(1, 2, 2, 1, 1, 2),
            options,
        )
        .unwrap();
        let actual = layer
            .forward_self_attention(&hidden_states, 1, 2, options)
            .unwrap();

        approx_eq_slice(&actual, &expected);
        assert_eq!(layer.kernel(), GROUPED_QUERY_ATTENTION);
    }

    /// Verifies that causal self-attention with identity projections reproduces the causal kernel output.
    ///
    /// This catches incorrect causal-mask propagation through `AttentionOptions`.
    #[test]
    fn causal_attention_matches_kernel_when_projections_are_identity() {
        let layer = AttentionLayer::new(
            2,
            1,
            1,
            2,
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
        )
        .unwrap();
        let hidden_states = [1.0, 0.0, 0.0, 1.0];
        let options = AttentionOptions {
            causal: true,
            softmax_scale: Some(1.0),
            ..AttentionOptions::default()
        };

        let expected = scaled_dot_product_attention_host(
            &hidden_states,
            &hidden_states,
            &hidden_states,
            AttentionShape::new(1, 2, 2, 1, 1, 2),
            options,
        )
        .unwrap();
        let actual = layer
            .forward_self_attention(&hidden_states, 1, 2, options)
            .unwrap();

        approx_eq_slice(&actual, &expected);
    }

    /// Verifies that construction rejects an output projection with wrong input dimensions.
    ///
    /// This catches weakened projection-shape validation in `AttentionLayer::new`.
    #[test]
    fn rejects_projection_shape_mismatch() {
        let error = AttentionLayer::new(
            2,
            1,
            1,
            2,
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
            LinearProjection::new_dense_f32(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], None)
                .unwrap(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            AttentionLayerError::ProjectionShapeMismatch {
                projection: "output_projection",
                expected_input_dim: 2,
                actual_input_dim: 3,
                expected_output_dim: 2,
                actual_output_dim: 2,
            }
        );
    }

    /// Verifies that forward rejects a query_states slice shorter than expected.
    ///
    /// This catches missing or incorrect hidden-state length validation.
    #[test]
    fn rejects_query_state_length_mismatch() {
        let layer = AttentionLayer::new(
            2,
            1,
            1,
            2,
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
        )
        .unwrap();

        let error = layer
            .forward(
                &[1.0, 0.0],
                &[1.0, 0.0, 0.0, 1.0],
                AttentionForwardShape::new(1, 2, 2),
                AttentionOptions::default(),
            )
            .unwrap_err();

        assert_eq!(
            error,
            AttentionLayerError::LengthMismatch {
                argument: "query_states",
                expected: 4,
                actual: 2,
            }
        );
    }
}
