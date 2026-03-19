use crate::linear::{LinearError, LinearProjection, LinearShape, DENSE_F32_HOST};
use crate::{LayerStub, MlpError, MlpLayer, MlpShape};
use nemotron_kernels::moe_routing::{moe_route_host, MoeRoutingError, MoeRoutingShape};
use nemotron_kernels::tensor::GpuTensor;
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "moe",
    summary: "Host-fallback Mixture-of-Experts layer with sigmoid_host top-k routing.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoeBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeKernel {
    pub name: &'static str,
    pub backend: MoeBackend,
}

pub const MOE_DENSE_HOST: MoeKernel = MoeKernel {
    name: "moe_dense_host",
    backend: MoeBackend::HostFallback,
};

pub fn supported_moe_kernels() -> [MoeKernel; 1] {
    [MOE_DENSE_HOST]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeShape {
    pub hidden_dim: usize,
    pub expert_count: usize,
    pub top_k: usize,
}

impl MoeShape {
    pub const fn new(hidden_dim: usize, expert_count: usize, top_k: usize) -> Self {
        Self {
            hidden_dim,
            expert_count,
            top_k,
        }
    }

    pub const fn input_len(self, row_count: usize) -> usize {
        row_count * self.hidden_dim
    }

    pub const fn router_shape(self) -> LinearShape {
        LinearShape::new(self.hidden_dim, self.expert_count)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeLayer {
    shape: MoeShape,
    router: LinearProjection,
    experts: Vec<MlpLayer>,
    shared_expert: Option<MlpLayer>,
}

impl MoeLayer {
    pub fn new(
        shape: MoeShape,
        router: LinearProjection,
        experts: Vec<MlpLayer>,
        shared_expert: Option<MlpLayer>,
    ) -> Result<Self, MoeError> {
        validate_shape(shape)?;

        if router.shape() != shape.router_shape() {
            return Err(MoeError::RouterShapeMismatch {
                expected: shape.router_shape(),
                actual: router.shape(),
            });
        }

        if experts.len() != shape.expert_count {
            return Err(MoeError::ExpertCountMismatch {
                expected: shape.expert_count,
                actual: experts.len(),
            });
        }

        for (expert_index, expert) in experts.iter().enumerate() {
            validate_expert_shape("expert", expert.shape(), shape.hidden_dim).map_err(
                |source| MoeError::ExpertShapeMismatch {
                    expert_index,
                    source,
                },
            )?;
        }

        if let Some(shared_expert) = shared_expert.as_ref() {
            validate_expert_shape("shared_expert", shared_expert.shape(), shape.hidden_dim)
                .map_err(MoeError::SharedExpertShapeMismatch)?;
        }

        Ok(Self {
            shape,
            router,
            experts,
            shared_expert,
        })
    }

    pub const fn shape(&self) -> MoeShape {
        self.shape
    }

    pub fn router(&self) -> &LinearProjection {
        &self.router
    }

    pub fn experts(&self) -> &[MlpLayer] {
        &self.experts
    }

    pub fn shared_expert(&self) -> Option<&MlpLayer> {
        self.shared_expert.as_ref()
    }

    pub fn kernel(&self) -> Option<MoeKernel> {
        if self.router.kernel() != Some(DENSE_F32_HOST) {
            return None;
        }

        if self.experts.iter().any(|expert| expert.kernel().is_none()) {
            return None;
        }

        if self
            .shared_expert
            .as_ref()
            .is_some_and(|expert| expert.kernel().is_none())
        {
            return None;
        }

        Some(MOE_DENSE_HOST)
    }

    pub fn forward(&self, input: &[f32], row_count: usize) -> Result<Vec<f32>, MoeError> {
        let mut output = vec![0.0; self.shape.input_len(row_count)];
        self.forward_into(input, row_count, &mut output)?;
        Ok(output)
    }

    pub fn forward_into(
        &self,
        input: &[f32],
        row_count: usize,
        output: &mut [f32],
    ) -> Result<(), MoeError> {
        if row_count == 0 {
            return Err(MoeError::InvalidRowCount(row_count));
        }

        let expected_input_len = self.shape.input_len(row_count);
        if input.len() != expected_input_len {
            return Err(MoeError::LengthMismatch {
                argument: "input",
                expected: expected_input_len,
                actual: input.len(),
            });
        }

        if output.len() != expected_input_len {
            return Err(MoeError::LengthMismatch {
                argument: "output",
                expected: expected_input_len,
                actual: output.len(),
            });
        }

        if self.router.kernel() != Some(DENSE_F32_HOST) {
            return Err(MoeError::UnsupportedRouter);
        }

        let router_scores = self
            .router
            .project(input, row_count)
            .map_err(MoeError::Router)?;
        let routes = moe_route_host(
            &router_scores,
            MoeRoutingShape::new(row_count, self.shape.expert_count, self.shape.top_k),
        )
        .map_err(MoeError::Routing)?;

        output.fill(0.0);
        let mut expert_outputs = vec![None; self.shape.expert_count];
        let shared_output = if let Some(shared_expert) = self.shared_expert.as_ref() {
            Some(
                shared_expert
                    .forward(input, row_count)
                    .map_err(MoeError::SharedExpert)?,
            )
        } else {
            None
        };

        for token_index in 0..row_count {
            let token_start = token_index * self.shape.hidden_dim;
            let token_end = token_start + self.shape.hidden_dim;
            let output_row = &mut output[token_start..token_end];

            if let Some(shared_output) = shared_output.as_ref() {
                for (value, shared_value) in output_row
                    .iter_mut()
                    .zip(shared_output[token_start..token_end].iter().copied())
                {
                    *value += shared_value;
                }
            }

            for route_index in 0..self.shape.top_k {
                let route_offset = token_index * self.shape.top_k + route_index;
                let expert_index = routes.indices[route_offset];
                let weight = routes.weights[route_offset];

                if expert_outputs[expert_index].is_none() {
                    let result = self.experts[expert_index]
                        .forward(input, row_count)
                        .map_err(|source| MoeError::Expert {
                            expert_index,
                            source,
                        })?;
                    expert_outputs[expert_index] = Some(result);
                }
                let expert_output = expert_outputs[expert_index]
                    .as_ref()
                    .expect("expert output inserted immediately above");

                for (value, expert_value) in output_row
                    .iter_mut()
                    .zip(expert_output[token_start..token_end].iter().copied())
                {
                    *value += weight * expert_value;
                }
            }
        }

        Ok(())
    }

    /// Async GPU MoE forward. Delegates to host fallback via data transfer.
    pub async fn forward_gpu(
        &self,
        input: &GpuTensor,
        row_count: usize,
    ) -> Result<GpuTensor, MoeError> {
        let data = input
            .to_host_async()
            .await
            .map_err(|e| MoeError::DeviceError(e.to_string()))?;
        let result = self.forward(&data, row_count)?;
        GpuTensor::from_host_async(&result, &[row_count, self.shape.hidden_dim])
            .await
            .map_err(|e| MoeError::DeviceError(e.to_string()))
    }
}

fn validate_shape(shape: MoeShape) -> Result<(), MoeError> {
    if shape.hidden_dim == 0
        || shape.expert_count == 0
        || shape.top_k == 0
        || shape.top_k > shape.expert_count
    {
        return Err(MoeError::InvalidShape(shape));
    }

    Ok(())
}

fn validate_expert_shape(
    _name: &'static str,
    shape: MlpShape,
    hidden_dim: usize,
) -> Result<(), MlpShapeMismatch> {
    if shape.hidden_dim != hidden_dim {
        return Err(MlpShapeMismatch {
            expected_hidden_dim: hidden_dim,
            actual_hidden_dim: shape.hidden_dim,
        });
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlpShapeMismatch {
    pub expected_hidden_dim: usize,
    pub actual_hidden_dim: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MoeError {
    InvalidShape(MoeShape),
    InvalidRowCount(usize),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    RouterShapeMismatch {
        expected: LinearShape,
        actual: LinearShape,
    },
    ExpertCountMismatch {
        expected: usize,
        actual: usize,
    },
    ExpertShapeMismatch {
        expert_index: usize,
        source: MlpShapeMismatch,
    },
    SharedExpertShapeMismatch(MlpShapeMismatch),
    UnsupportedRouter,
    Router(LinearError),
    Routing(MoeRoutingError),
    Expert {
        expert_index: usize,
        source: MlpError,
    },
    SharedExpert(MlpError),
    DeviceError(String),
}

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(shape) => write!(
                f,
                "moe shape must be valid, got hidden_dim={}, expert_count={}, top_k={}",
                shape.hidden_dim, shape.expert_count, shape.top_k
            ),
            Self::InvalidRowCount(row_count) => write!(f, "row_count must be non-zero, got {row_count}"),
            Self::LengthMismatch {
                argument,
                expected,
                actual,
            } => write!(
                f,
                "{argument} length mismatch: expected {expected}, got {actual}"
            ),
            Self::RouterShapeMismatch { expected, actual } => write!(
                f,
                "router shape mismatch: expected input_dim={}, output_dim={}, got input_dim={}, output_dim={}",
                expected.input_dim, expected.output_dim, actual.input_dim, actual.output_dim
            ),
            Self::ExpertCountMismatch { expected, actual } => write!(
                f,
                "expert count mismatch: expected {expected}, got {actual}"
            ),
            Self::ExpertShapeMismatch { expert_index, source } => write!(
                f,
                "expert {expert_index} hidden size mismatch: expected {}, got {}",
                source.expected_hidden_dim, source.actual_hidden_dim
            ),
            Self::SharedExpertShapeMismatch(source) => write!(
                f,
                "shared expert hidden size mismatch: expected {}, got {}",
                source.expected_hidden_dim, source.actual_hidden_dim
            ),
            Self::UnsupportedRouter => write!(f, "router must use the dense host linear path"),
            Self::Router(source) => write!(f, "router projection failed: {source}"),
            Self::Routing(source) => write!(f, "routing failed: {source:?}"),
            Self::Expert {
                expert_index,
                source,
            } => write!(f, "expert {expert_index} failed: {source}"),
            Self::SharedExpert(source) => write!(f, "shared expert failed: {source}"),
            Self::DeviceError(msg) => write!(f, "device error: {msg}"),
        }
    }
}

impl Error for MoeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use nemotron_kernels::moe_routing::MOE_SIGMOID_TOPK;

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

    fn scaling_mlp(scale: f32) -> MlpLayer {
        MlpLayer::new_dense_relu2(1, 1, vec![1.0], None, vec![scale], None).unwrap()
    }

    /// Verifies the full MoE path: router → sigmoid_host top-k → expert execution → weighted combination + shared expert.
    ///
    /// This catches incorrect routing weight application, expert selection, or shared-expert addition.
    #[test]
    fn combines_top_k_expert_outputs_and_shared_expert() {
        let layer = MoeLayer::new(
            MoeShape::new(1, 3, 2),
            dense_projection(1, 3, vec![0.0, 0.0, 0.0], Some(vec![0.2, 2.0, 1.0])),
            vec![scaling_mlp(1.0), scaling_mlp(2.0), scaling_mlp(3.0)],
            Some(scaling_mlp(0.5)),
        )
        .unwrap();

        let output = layer.forward(&[2.0], 1).unwrap();

        let expected_weight_1 = 0.880797;
        let expected_weight_2 = 0.7310586;
        let expected = 0.5 * 4.0 + expected_weight_1 * 8.0 + expected_weight_2 * 12.0;
        approx_eq_slice(&output, &[expected]);
        assert_eq!(layer.kernel(), Some(MOE_DENSE_HOST));
        assert_eq!(MOE_SIGMOID_TOPK.name, "moe_sigmoid_topk");
    }

    /// Verifies that forward_into overwrites a caller-provided buffer with MoE results.
    ///
    /// This catches buffer reuse bugs or missing `output.fill(0.0)` before accumulation.
    #[test]
    fn writes_existing_output_buffer() {
        let layer = MoeLayer::new(
            MoeShape::new(1, 2, 1),
            dense_projection(1, 2, vec![0.0, 0.0], Some(vec![3.0, 1.0])),
            vec![scaling_mlp(1.0), scaling_mlp(2.0)],
            None,
        )
        .unwrap();
        let mut output = [-1.0; 1];

        layer.forward_into(&[2.0], 1, &mut output).unwrap();

        approx_eq_slice(&output, &[0.95257413 * 4.0]);
    }

    /// Verifies that construction rejects fewer experts than `expert_count` specifies.
    ///
    /// This catches weakened expert-count validation in `MoeLayer::new`.
    #[test]
    fn rejects_expert_count_mismatch() {
        let error = MoeLayer::new(
            MoeShape::new(1, 2, 1),
            dense_projection(1, 2, vec![1.0, 0.0], None),
            vec![scaling_mlp(1.0)],
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            MoeError::ExpertCountMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that construction rejects a router whose output dim doesn't match expert_count.
    ///
    /// This catches weakened router-shape validation in `MoeLayer::new`.
    #[test]
    fn rejects_router_shape_mismatch() {
        let error = MoeLayer::new(
            MoeShape::new(1, 2, 1),
            dense_projection(1, 3, vec![1.0, 0.0, 0.0], None),
            vec![scaling_mlp(1.0), scaling_mlp(2.0)],
            None,
        )
        .unwrap_err();

        assert_eq!(
            error,
            MoeError::RouterShapeMismatch {
                expected: LinearShape::new(1, 2),
                actual: LinearShape::new(1, 3),
            }
        );
    }

    /// Verifies that the GPU MoE forward path matches the host-fallback output.
    ///
    /// This catches regressions in the async GPU data transfer path for MoE layers.
    #[tokio::test]
    async fn gpu_moe_forward_matches_host() {
        use nemotron_kernels::tensor::GpuTensor;
        let layer = MoeLayer::new(
            MoeShape::new(1, 3, 2),
            dense_projection(1, 3, vec![0.0, 0.0, 0.0], Some(vec![0.2, 2.0, 1.0])),
            vec![scaling_mlp(1.0), scaling_mlp(2.0), scaling_mlp(3.0)],
            Some(scaling_mlp(0.5)),
        )
        .unwrap();
        let input = [1.0_f32];

        let host_result = layer.forward(&input, 1).unwrap();

        let gpu_input = GpuTensor::from_host(&input, &[1, 1]).unwrap();
        let gpu_result = layer.forward_gpu(&gpu_input, 1).await.unwrap();
        let gpu_host = gpu_result.to_host();

        approx_eq_slice(&gpu_host, &host_result);
    }
}
