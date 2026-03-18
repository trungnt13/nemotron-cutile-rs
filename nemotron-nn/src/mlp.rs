use crate::linear::{LinearError, LinearProjection, LinearShape, LinearWeightKind, DENSE_F32_HOST};
use crate::LayerStub;
use nemotron_kernels::activations::{relu2_in_place, ActivationKernel, RELU2};
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "mlp",
    summary: "Dense Nemotron MLP layer with host-fallback ReLU² activation.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlpBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpKernel {
    pub name: &'static str,
    pub backend: MlpBackend,
    pub activation: ActivationKernel,
}

pub const DENSE_RELU2_HOST: MlpKernel = MlpKernel {
    name: "mlp_dense_relu2",
    backend: MlpBackend::HostFallback,
    activation: RELU2,
};

pub fn supported_mlp_kernels() -> [MlpKernel; 1] {
    [DENSE_RELU2_HOST]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlpShape {
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

impl MlpShape {
    pub const fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            hidden_dim,
            intermediate_dim,
        }
    }

    pub const fn input_len(self, row_count: usize) -> usize {
        row_count * self.hidden_dim
    }

    pub const fn output_len(self, row_count: usize) -> usize {
        row_count * self.hidden_dim
    }

    pub const fn activated_len(self, row_count: usize) -> usize {
        row_count * self.intermediate_dim
    }

    pub const fn up_projection_shape(self) -> LinearShape {
        LinearShape::new(self.hidden_dim, self.intermediate_dim)
    }

    pub const fn down_projection_shape(self) -> LinearShape {
        LinearShape::new(self.intermediate_dim, self.hidden_dim)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MlpLayer {
    shape: MlpShape,
    up_projection: LinearProjection,
    down_projection: LinearProjection,
}

impl MlpLayer {
    pub fn new_dense_relu2(
        hidden_dim: usize,
        intermediate_dim: usize,
        up_weights: Vec<f32>,
        up_bias: Option<Vec<f32>>,
        down_weights: Vec<f32>,
        down_bias: Option<Vec<f32>>,
    ) -> Result<Self, MlpError> {
        let shape = MlpShape::new(hidden_dim, intermediate_dim);
        let up_projection =
            LinearProjection::new_dense_f32(hidden_dim, intermediate_dim, up_weights, up_bias)
                .map_err(MlpError::UpProjection)?;
        let down_projection =
            LinearProjection::new_dense_f32(intermediate_dim, hidden_dim, down_weights, down_bias)
                .map_err(MlpError::DownProjection)?;

        Self::new(shape, up_projection, down_projection)
    }

    pub fn new(
        shape: MlpShape,
        up_projection: LinearProjection,
        down_projection: LinearProjection,
    ) -> Result<Self, MlpError> {
        validate_mlp_shape(shape)?;

        let expected_up = shape.up_projection_shape();
        if up_projection.shape() != expected_up {
            return Err(MlpError::ProjectionShapeMismatch {
                projection: "up_projection",
                expected: expected_up,
                actual: up_projection.shape(),
            });
        }

        let expected_down = shape.down_projection_shape();
        if down_projection.shape() != expected_down {
            return Err(MlpError::ProjectionShapeMismatch {
                projection: "down_projection",
                expected: expected_down,
                actual: down_projection.shape(),
            });
        }

        Ok(Self {
            shape,
            up_projection,
            down_projection,
        })
    }

    pub const fn shape(&self) -> MlpShape {
        self.shape
    }

    pub fn up_projection(&self) -> &LinearProjection {
        &self.up_projection
    }

    pub fn down_projection(&self) -> &LinearProjection {
        &self.down_projection
    }

    pub fn kernel(&self) -> Option<MlpKernel> {
        match (self.up_projection.kernel(), self.down_projection.kernel()) {
            (Some(up), Some(down)) if up == DENSE_F32_HOST && down == DENSE_F32_HOST => {
                Some(DENSE_RELU2_HOST)
            }
            _ => None,
        }
    }

    pub fn forward(&self, input: &[f32], row_count: usize) -> Result<Vec<f32>, MlpError> {
        let mut output = vec![0.0; self.shape.output_len(row_count)];
        self.forward_into(input, row_count, &mut output)?;
        Ok(output)
    }

    pub fn forward_into(
        &self,
        input: &[f32],
        row_count: usize,
        output: &mut [f32],
    ) -> Result<(), MlpError> {
        if row_count == 0 {
            return Err(MlpError::InvalidRowCount(row_count));
        }

        let expected_input_len = self.shape.input_len(row_count);
        if input.len() != expected_input_len {
            return Err(MlpError::LengthMismatch {
                argument: "input",
                expected: expected_input_len,
                actual: input.len(),
            });
        }

        let expected_output_len = self.shape.output_len(row_count);
        if output.len() != expected_output_len {
            return Err(MlpError::LengthMismatch {
                argument: "output",
                expected: expected_output_len,
                actual: output.len(),
            });
        }

        if self.up_projection.kernel() != Some(DENSE_F32_HOST) {
            return Err(MlpError::UnsupportedProjection {
                projection: "up_projection",
                backend: MlpBackend::HostFallback,
                weight_kind: self.up_projection.weights().kind(),
            });
        }

        if self.down_projection.kernel() != Some(DENSE_F32_HOST) {
            return Err(MlpError::UnsupportedProjection {
                projection: "down_projection",
                backend: MlpBackend::HostFallback,
                weight_kind: self.down_projection.weights().kind(),
            });
        }

        let mut activated = self
            .up_projection
            .project(input, row_count)
            .map_err(MlpError::UpProjection)?;
        relu2_in_place(&mut activated);
        self.down_projection
            .project_into(&activated, row_count, output)
            .map_err(MlpError::DownProjection)?;

        Ok(())
    }
}

fn validate_mlp_shape(shape: MlpShape) -> Result<(), MlpError> {
    if shape.hidden_dim == 0 || shape.intermediate_dim == 0 {
        return Err(MlpError::InvalidShape(shape));
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum MlpError {
    InvalidShape(MlpShape),
    InvalidRowCount(usize),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    ProjectionShapeMismatch {
        projection: &'static str,
        expected: LinearShape,
        actual: LinearShape,
    },
    UnsupportedProjection {
        projection: &'static str,
        backend: MlpBackend,
        weight_kind: LinearWeightKind,
    },
    UpProjection(LinearError),
    DownProjection(LinearError),
}

impl fmt::Display for MlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(shape) => write!(
                f,
                "mlp shape must be non-zero, got hidden_dim={}, intermediate_dim={}",
                shape.hidden_dim, shape.intermediate_dim
            ),
            Self::InvalidRowCount(row_count) => {
                write!(f, "row_count must be non-zero, got {row_count}")
            }
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
                expected,
                actual,
            } => write!(
                f,
                "{projection} shape mismatch: expected input_dim={}, output_dim={}, got input_dim={}, output_dim={}",
                expected.input_dim,
                expected.output_dim,
                actual.input_dim,
                actual.output_dim
            ),
            Self::UnsupportedProjection {
                projection,
                backend,
                weight_kind,
            } => write!(
                f,
                "{projection} weights {:?} are not supported on {:?} yet",
                weight_kind, backend
            ),
            Self::UpProjection(error) => write!(f, "up projection failed: {error}"),
            Self::DownProjection(error) => write!(f, "down projection failed: {error}"),
        }
    }
}

impl Error for MlpError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::LinearProjection;
    use nemotron_kernels::quantize::{quantize_int4, Int4QuantizationParams};

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

    /// Verifies that the supported kernel list contains only the dense-ReLU²-host path.
    ///
    /// This catches accidental removal or duplication of MLP kernel entries.
    #[test]
    fn reports_dense_relu2_host_kernel_for_now() {
        assert_eq!(
            supported_mlp_kernels(),
            [MlpKernel {
                name: "mlp_dense_relu2",
                backend: MlpBackend::HostFallback,
                activation: RELU2,
            }]
        );
    }

    /// Verifies the full MLP path: up-project → ReLU² → down-project + bias for two rows.
    ///
    /// This catches incorrect activation placement, wrong projection ordering, or bias errors.
    #[test]
    fn runs_dense_nemotron_style_path() {
        let layer = MlpLayer::new_dense_relu2(
            2,
            3,
            vec![1.0, -1.0, 0.5, 0.5, 1.0, 1.0],
            Some(vec![0.0, 1.0, -0.5]),
            vec![1.0, 2.0, -1.0, 0.0, 0.5, -0.5],
            Some(vec![0.25, -0.25]),
        )
        .unwrap();

        let output = layer.forward(&[2.0, -1.0, 0.0, 3.0], 2).unwrap();

        assert_eq!(layer.shape(), MlpShape::new(2, 3));
        assert_eq!(layer.kernel(), Some(DENSE_RELU2_HOST));
        approx_eq_slice(&output, &[2.5, 4.25, -10.375, 1.125]);
    }

    /// Verifies that forward_into overwrites a caller-provided buffer with ReLU²-activated results.
    ///
    /// This catches buffer reuse bugs where stale data leaks into the output.
    #[test]
    fn forward_into_writes_existing_output() {
        let layer = MlpLayer::new_dense_relu2(
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            None,
            vec![1.0, 0.0, 0.0, 1.0],
            None,
        )
        .unwrap();
        let mut output = [-1.0; 4];

        layer
            .forward_into(&[-2.0, 3.0, 4.0, -1.0], 2, &mut output)
            .unwrap();

        approx_eq_slice(&output, &[0.0, 9.0, 16.0, 0.0]);
    }

    /// Verifies that construction rejects an up-projection whose shape doesn't match `MlpShape`.
    ///
    /// This catches weakened shape validation in `MlpLayer::new`.
    #[test]
    fn rejects_projection_shape_mismatches() {
        let up = LinearProjection::new_dense_f32(2, 4, vec![0.0; 8], None).unwrap();
        let down = LinearProjection::new_dense_f32(3, 2, vec![0.0; 6], None).unwrap();

        let error = MlpLayer::new(MlpShape::new(2, 3), up, down).unwrap_err();

        assert_eq!(
            error,
            MlpError::ProjectionShapeMismatch {
                projection: "up_projection",
                expected: LinearShape::new(2, 3),
                actual: LinearShape::new(2, 4),
            }
        );
    }

    /// Verifies that forward rejects int4-affine projections that lack a host-path kernel.
    ///
    /// This catches premature removal of the unsupported-weight guard.
    #[test]
    fn rejects_unsupported_non_dense_projection() {
        let params = Int4QuantizationParams::new(0.5, 8);
        let packed = quantize_int4(&[1.0, 0.0, 0.0, 1.0], params).unwrap();
        let up = LinearProjection::new_int4_affine(2, 2, packed, params, None).unwrap();
        let down = LinearProjection::new_dense_f32(2, 2, vec![1.0, 0.0, 0.0, 1.0], None).unwrap();
        let layer = MlpLayer::new(MlpShape::new(2, 2), up, down).unwrap();

        let error = layer.forward(&[1.0, 2.0], 1).unwrap_err();

        assert_eq!(
            error,
            MlpError::UnsupportedProjection {
                projection: "up_projection",
                backend: MlpBackend::HostFallback,
                weight_kind: LinearWeightKind::Int4Affine,
            }
        );
    }

    /// Verifies that forward rejects zero row_count, wrong input length, and wrong output length.
    ///
    /// This catches missing guards in `forward` and `forward_into` runtime validation.
    #[test]
    fn rejects_invalid_runtime_shapes() {
        let layer = MlpLayer::new_dense_relu2(
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            None,
            vec![1.0, 0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        assert_eq!(
            layer.forward(&[], 0).unwrap_err(),
            MlpError::InvalidRowCount(0)
        );
        assert_eq!(
            layer.forward(&[1.0, 2.0, 3.0], 2).unwrap_err(),
            MlpError::LengthMismatch {
                argument: "input",
                expected: 4,
                actual: 3,
            }
        );

        let mut output = [0.0; 3];
        assert_eq!(
            layer
                .forward_into(&[1.0, 2.0, 3.0, 4.0], 2, &mut output)
                .unwrap_err(),
            MlpError::LengthMismatch {
                argument: "output",
                expected: 4,
                actual: 3,
            }
        );
    }
}
