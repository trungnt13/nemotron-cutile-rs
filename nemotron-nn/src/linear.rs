use crate::LayerStub;
use nemotron_kernels::gemm::{gemm_into, GemmError, GemmShape};
use nemotron_kernels::quantize::{
    dequantize_int4, packed_int4_len, validate_int4_params, Int4QuantizationParams, QuantizeError,
};
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "linear",
    summary: "Linear projection layer with a dense/f32 host path and explicit future hooks.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearWeightKind {
    DenseF32,
    Int4Affine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearKernel {
    pub name: &'static str,
    pub backend: LinearBackend,
    pub weight_kind: LinearWeightKind,
}

pub const DENSE_F32_HOST: LinearKernel = LinearKernel {
    name: "linear_dense_f32",
    backend: LinearBackend::HostFallback,
    weight_kind: LinearWeightKind::DenseF32,
};

pub fn supported_linear_kernels() -> [LinearKernel; 1] {
    [DENSE_F32_HOST]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearShape {
    pub input_dim: usize,
    pub output_dim: usize,
}

impl LinearShape {
    pub const fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
        }
    }

    pub const fn weight_len(self) -> usize {
        self.input_dim * self.output_dim
    }

    pub const fn input_len(self, row_count: usize) -> usize {
        row_count * self.input_dim
    }

    pub const fn output_len(self, row_count: usize) -> usize {
        row_count * self.output_dim
    }

    const fn gemm_shape(self, row_count: usize) -> GemmShape {
        GemmShape::new(row_count, self.input_dim, self.output_dim)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseLinearWeights {
    pub values: Vec<f32>,
}

impl DenseLinearWeights {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    fn validate(&self, shape: LinearShape) -> Result<(), LinearError> {
        if self.values.len() != shape.weight_len() {
            return Err(LinearError::LengthMismatch {
                argument: "weights",
                expected: shape.weight_len(),
                actual: self.values.len(),
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Int4LinearWeights {
    pub packed_values: Vec<u8>,
    pub params: Int4QuantizationParams,
}

impl Int4LinearWeights {
    pub fn new(packed_values: Vec<u8>, params: Int4QuantizationParams) -> Self {
        Self {
            packed_values,
            params,
        }
    }

    fn validate(&self, shape: LinearShape) -> Result<(), LinearError> {
        let expected = packed_int4_len(shape.weight_len());
        if self.packed_values.len() != expected {
            return Err(LinearError::LengthMismatch {
                argument: "packed_weights",
                expected,
                actual: self.packed_values.len(),
            });
        }

        validate_int4_params(self.params).map_err(LinearError::Quantize)
    }

    pub fn materialize_dense(&self, shape: LinearShape) -> Result<DenseLinearWeights, LinearError> {
        self.validate(shape)?;
        let values = dequantize_int4(&self.packed_values, shape.weight_len(), self.params)
            .map_err(LinearError::Quantize)?;
        Ok(DenseLinearWeights { values })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LinearWeights {
    DenseF32(DenseLinearWeights),
    Int4Affine(Int4LinearWeights),
}

impl LinearWeights {
    pub fn kind(&self) -> LinearWeightKind {
        match self {
            Self::DenseF32(_) => LinearWeightKind::DenseF32,
            Self::Int4Affine(_) => LinearWeightKind::Int4Affine,
        }
    }

    pub fn supported_kernel(&self) -> Option<LinearKernel> {
        match self {
            Self::DenseF32(_) => Some(DENSE_F32_HOST),
            Self::Int4Affine(_) => None,
        }
    }

    fn validate(&self, shape: LinearShape) -> Result<(), LinearError> {
        match self {
            Self::DenseF32(weights) => weights.validate(shape),
            Self::Int4Affine(weights) => weights.validate(shape),
        }
    }

    pub fn materialize_dense(&self, shape: LinearShape) -> Result<DenseLinearWeights, LinearError> {
        match self {
            Self::DenseF32(weights) => {
                weights.validate(shape)?;
                Ok(weights.clone())
            }
            Self::Int4Affine(weights) => weights.materialize_dense(shape),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearProjection {
    shape: LinearShape,
    weights: LinearWeights,
    bias: Option<Vec<f32>>,
}

impl LinearProjection {
    pub fn new_dense_f32(
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, LinearError> {
        Self::new(
            input_dim,
            output_dim,
            LinearWeights::DenseF32(DenseLinearWeights::new(weights)),
            bias,
        )
    }

    pub fn new_int4_affine(
        input_dim: usize,
        output_dim: usize,
        packed_weights: Vec<u8>,
        params: Int4QuantizationParams,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, LinearError> {
        Self::new(
            input_dim,
            output_dim,
            LinearWeights::Int4Affine(Int4LinearWeights::new(packed_weights, params)),
            bias,
        )
    }

    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weights: LinearWeights,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, LinearError> {
        Self::from_shape(LinearShape::new(input_dim, output_dim), weights, bias)
    }

    pub fn from_shape(
        shape: LinearShape,
        weights: LinearWeights,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, LinearError> {
        validate_projection_shape(shape)?;

        if let Some(bias_values) = bias.as_ref() {
            if bias_values.len() != shape.output_dim {
                return Err(LinearError::LengthMismatch {
                    argument: "bias",
                    expected: shape.output_dim,
                    actual: bias_values.len(),
                });
            }
        }

        weights.validate(shape)?;

        Ok(Self {
            shape,
            weights,
            bias,
        })
    }

    pub const fn shape(&self) -> LinearShape {
        self.shape
    }

    pub const fn input_dim(&self) -> usize {
        self.shape.input_dim
    }

    pub const fn output_dim(&self) -> usize {
        self.shape.output_dim
    }

    pub fn weights(&self) -> &LinearWeights {
        &self.weights
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    pub fn kernel(&self) -> Option<LinearKernel> {
        self.weights.supported_kernel()
    }

    pub fn materialize_dense_weights(&self) -> Result<DenseLinearWeights, LinearError> {
        self.weights.materialize_dense(self.shape)
    }

    pub fn project(&self, input: &[f32], row_count: usize) -> Result<Vec<f32>, LinearError> {
        if row_count == 0 {
            return Err(LinearError::InvalidRowCount(row_count));
        }

        let mut output = vec![0.0; self.shape.output_len(row_count)];
        self.project_into(input, row_count, &mut output)?;
        Ok(output)
    }

    pub fn project_into(
        &self,
        input: &[f32],
        row_count: usize,
        output: &mut [f32],
    ) -> Result<(), LinearError> {
        if row_count == 0 {
            return Err(LinearError::InvalidRowCount(row_count));
        }

        let expected_input_len = self.shape.input_len(row_count);
        if input.len() != expected_input_len {
            return Err(LinearError::LengthMismatch {
                argument: "input",
                expected: expected_input_len,
                actual: input.len(),
            });
        }

        let expected_output_len = self.shape.output_len(row_count);
        if output.len() != expected_output_len {
            return Err(LinearError::LengthMismatch {
                argument: "output",
                expected: expected_output_len,
                actual: output.len(),
            });
        }

        let dense_weights = match &self.weights {
            LinearWeights::DenseF32(weights) => {
                weights.validate(self.shape)?;
                weights.values.as_slice()
            }
            LinearWeights::Int4Affine(_) => {
                return Err(LinearError::UnsupportedWeights {
                    backend: LinearBackend::HostFallback,
                    weight_kind: self.weights.kind(),
                });
            }
        };

        // Weights are stored row-major as [input_dim, output_dim].
        gemm_into(
            input,
            dense_weights,
            self.shape.gemm_shape(row_count),
            output,
        )
        .map_err(LinearError::Gemm)?;

        if let Some(bias) = self.bias.as_deref() {
            for row in output.chunks_exact_mut(self.shape.output_dim) {
                for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
                    *value += bias_value;
                }
            }
        }

        Ok(())
    }
}

fn validate_projection_shape(shape: LinearShape) -> Result<(), LinearError> {
    if shape.input_dim == 0 || shape.output_dim == 0 {
        return Err(LinearError::InvalidShape(shape));
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum LinearError {
    InvalidShape(LinearShape),
    InvalidRowCount(usize),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    UnsupportedWeights {
        backend: LinearBackend,
        weight_kind: LinearWeightKind,
    },
    Gemm(GemmError),
    Quantize(QuantizeError),
}

impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(shape) => write!(
                f,
                "linear projection shape must be non-zero, got input_dim={}, output_dim={}",
                shape.input_dim, shape.output_dim
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
            Self::UnsupportedWeights {
                backend,
                weight_kind,
            } => write!(
                f,
                "linear weights {:?} are not supported on {:?} yet",
                weight_kind, backend
            ),
            Self::Gemm(error) => write!(f, "gemm failed: {error:?}"),
            Self::Quantize(error) => write!(f, "quantization failed: {error:?}"),
        }
    }
}

impl Error for LinearError {}

#[cfg(test)]
mod tests {
    use super::*;
    use nemotron_kernels::quantize::quantize_int4;

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
    fn reports_dense_host_kernel_for_now() {
        assert_eq!(
            supported_linear_kernels(),
            [LinearKernel {
                name: "linear_dense_f32",
                backend: LinearBackend::HostFallback,
                weight_kind: LinearWeightKind::DenseF32,
            }]
        );
    }

    #[test]
    fn projects_dense_weights_with_bias() {
        let layer = LinearProjection::new_dense_f32(
            2,
            2,
            vec![0.5, 1.0, -0.5, 0.0],
            Some(vec![0.25, -0.25]),
        )
        .unwrap();

        let output = layer.project(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();

        assert_eq!(layer.shape(), LinearShape::new(2, 2));
        assert_eq!(layer.kernel(), Some(DENSE_F32_HOST));
        approx_eq_slice(&output, &[-0.25, 0.75, -0.25, 2.75]);
    }

    #[test]
    fn project_into_writes_existing_output_without_bias() {
        let layer = LinearProjection::new_dense_f32(2, 2, vec![1.0, 0.0, 0.0, 1.0], None).unwrap();
        let mut output = [-1.0; 4];

        layer
            .project_into(&[3.0, 1.0, 4.0, 1.0], 2, &mut output)
            .unwrap();

        approx_eq_slice(&output, &[3.0, 1.0, 4.0, 1.0]);
    }

    #[test]
    fn materializes_int4_weights_via_quantize_kernel() {
        let params = Int4QuantizationParams::new(0.5, 8);
        let packed = quantize_int4(&[0.5, 1.0, -0.5, 0.0], params).unwrap();
        let layer = LinearProjection::new_int4_affine(2, 2, packed, params, None).unwrap();

        let dense = layer.materialize_dense_weights().unwrap();

        assert_eq!(layer.kernel(), None);
        approx_eq_slice(&dense.values, &[0.5, 1.0, -0.5, 0.0]);
    }

    #[test]
    fn rejects_projecting_unsupported_int4_weights() {
        let params = Int4QuantizationParams::new(0.5, 8);
        let packed = quantize_int4(&[0.5, 1.0, -0.5, 0.0], params).unwrap();
        let layer = LinearProjection::new_int4_affine(2, 2, packed, params, None).unwrap();

        let error = layer.project(&[1.0, 2.0, 3.0, 4.0], 2).unwrap_err();

        assert_eq!(
            error,
            LinearError::UnsupportedWeights {
                backend: LinearBackend::HostFallback,
                weight_kind: LinearWeightKind::Int4Affine,
            }
        );
    }

    #[test]
    fn rejects_invalid_projection_metadata() {
        let error = LinearProjection::new_dense_f32(0, 2, vec![], None).unwrap_err();
        assert_eq!(error, LinearError::InvalidShape(LinearShape::new(0, 2)));

        let error = LinearProjection::new_dense_f32(2, 2, vec![1.0, 2.0], None).unwrap_err();
        assert_eq!(
            error,
            LinearError::LengthMismatch {
                argument: "weights",
                expected: 4,
                actual: 2,
            }
        );

        let error =
            LinearProjection::new_dense_f32(2, 2, vec![1.0, 2.0, 3.0, 4.0], Some(vec![1.0]))
                .unwrap_err();
        assert_eq!(
            error,
            LinearError::LengthMismatch {
                argument: "bias",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_invalid_quantized_weight_metadata() {
        let params = Int4QuantizationParams::new(0.0, 8);
        let error =
            LinearProjection::new_int4_affine(2, 2, vec![0x00, 0x00], params, None).unwrap_err();

        assert_eq!(
            error,
            LinearError::Quantize(QuantizeError::InvalidParams(params))
        );

        let params = Int4QuantizationParams::new(0.5, 8);
        let error = LinearProjection::new_int4_affine(2, 2, vec![0x00], params, None).unwrap_err();
        assert_eq!(
            error,
            LinearError::LengthMismatch {
                argument: "packed_weights",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_invalid_runtime_shapes() {
        let layer = LinearProjection::new_dense_f32(2, 2, vec![1.0, 0.0, 0.0, 1.0], None).unwrap();

        let error = layer.project(&[], 0).unwrap_err();
        assert_eq!(error, LinearError::InvalidRowCount(0));

        let error = layer.project(&[1.0, 2.0, 3.0], 2).unwrap_err();
        assert_eq!(
            error,
            LinearError::LengthMismatch {
                argument: "input",
                expected: 4,
                actual: 3,
            }
        );

        let mut output = [0.0; 3];
        let error = layer
            .project_into(&[1.0, 2.0, 3.0, 4.0], 2, &mut output)
            .unwrap_err();
        assert_eq!(
            error,
            LinearError::LengthMismatch {
                argument: "output",
                expected: 4,
                actual: 3,
            }
        );
    }
}
