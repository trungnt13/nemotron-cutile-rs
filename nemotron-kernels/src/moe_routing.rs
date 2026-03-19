use crate::tensor::{GpuTensor, TensorError};
use crate::{activations::sigmoid_scalar, KernelStub};

pub const SPEC: KernelStub = KernelStub {
    name: "moe_routing",
    summary: "Top-k expert routing kernels with sigmoid_host scoring.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoeRoutingBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeRoutingKernel {
    pub name: &'static str,
    pub backend: MoeRoutingBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeRoutingShape {
    pub token_count: usize,
    pub expert_count: usize,
    pub top_k: usize,
}

impl MoeRoutingShape {
    pub const fn new(token_count: usize, expert_count: usize, top_k: usize) -> Self {
        Self {
            token_count,
            expert_count,
            top_k,
        }
    }

    pub const fn score_len(self) -> usize {
        self.token_count * self.expert_count
    }

    pub const fn route_len(self) -> usize {
        self.token_count * self.top_k
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeTokenRoute {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeRoutingOutput {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

pub const MOE_SIGMOID_TOPK: MoeRoutingKernel = MoeRoutingKernel {
    name: "moe_sigmoid_topk",
    backend: MoeRoutingBackend::HostFallback,
};

pub fn supported_moe_routing_kernels() -> [MoeRoutingKernel; 1] {
    [MOE_SIGMOID_TOPK]
}

pub fn moe_route_token_host(
    scores: &[f32],
    top_k: usize,
) -> Result<MoeTokenRoute, MoeRoutingError> {
    let mut indices = vec![0; top_k];
    let mut weights = vec![0.0; top_k];
    moe_route_token_into_host(scores, top_k, &mut indices, &mut weights)?;
    Ok(MoeTokenRoute { indices, weights })
}

pub fn moe_route_token_into_host(
    scores: &[f32],
    top_k: usize,
    indices: &mut [usize],
    weights: &mut [f32],
) -> Result<(), MoeRoutingError> {
    let shape = MoeRoutingShape::new(1, scores.len(), top_k);
    validate_shape(shape)?;
    validate_route_outputs(top_k, indices, weights)?;
    select_top_k(scores, top_k, indices, weights);
    Ok(())
}

pub fn moe_route_host(
    scores: &[f32],
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    let mut indices = vec![0; shape.route_len()];
    let mut weights = vec![0.0; shape.route_len()];
    moe_route_into_host(scores, shape, &mut indices, &mut weights)?;
    Ok(MoeRoutingOutput { indices, weights })
}

pub fn moe_route_softmax_host(
    scores: &[f32],
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    let mut indices = vec![0; shape.route_len()];
    let mut weights = vec![0.0; shape.route_len()];
    moe_route_softmax_into_host(scores, shape, &mut indices, &mut weights)?;
    Ok(MoeRoutingOutput { indices, weights })
}

pub fn moe_route_into_host(
    scores: &[f32],
    shape: MoeRoutingShape,
    indices: &mut [usize],
    weights: &mut [f32],
) -> Result<(), MoeRoutingError> {
    validate_route(scores, shape, indices, weights)?;

    for token_index in 0..shape.token_count {
        let score_start = token_index * shape.expert_count;
        let score_end = score_start + shape.expert_count;
        let route_start = token_index * shape.top_k;
        let route_end = route_start + shape.top_k;

        select_top_k(
            &scores[score_start..score_end],
            shape.top_k,
            &mut indices[route_start..route_end],
            &mut weights[route_start..route_end],
        );
    }

    Ok(())
}

pub fn moe_route_softmax_into_host(
    scores: &[f32],
    shape: MoeRoutingShape,
    indices: &mut [usize],
    weights: &mut [f32],
) -> Result<(), MoeRoutingError> {
    validate_route(scores, shape, indices, weights)?;

    for token_index in 0..shape.token_count {
        let score_start = token_index * shape.expert_count;
        let score_end = score_start + shape.expert_count;
        let route_start = token_index * shape.top_k;
        let route_end = route_start + shape.top_k;

        select_top_k_softmax(
            &scores[score_start..score_end],
            shape.top_k,
            &mut indices[route_start..route_end],
            &mut weights[route_start..route_end],
        );
    }

    Ok(())
}

fn select_top_k(scores: &[f32], top_k: usize, indices: &mut [usize], weights: &mut [f32]) {
    let mut ranked_scores = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(expert_index, score)| (expert_index, sigmoid_scalar(score)))
        .collect::<Vec<_>>();

    ranked_scores.sort_by(|(left_index, left_weight), (right_index, right_weight)| {
        right_weight
            .total_cmp(left_weight)
            .then_with(|| left_index.cmp(right_index))
    });

    for (route_index, (expert_index, weight)) in ranked_scores.into_iter().take(top_k).enumerate() {
        indices[route_index] = expert_index;
        weights[route_index] = weight;
    }
}

fn select_top_k_softmax(scores: &[f32], top_k: usize, indices: &mut [usize], weights: &mut [f32]) {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut ranked_scores = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(expert_index, score)| (expert_index, (score - max_score).exp()))
        .collect::<Vec<_>>();
    let total: f32 = ranked_scores.iter().map(|(_, weight)| *weight).sum();
    for (_, weight) in &mut ranked_scores {
        *weight /= total;
    }

    ranked_scores.sort_by(|(left_index, left_weight), (right_index, right_weight)| {
        right_weight
            .total_cmp(left_weight)
            .then_with(|| left_index.cmp(right_index))
    });

    let selected_total: f32 = ranked_scores
        .iter()
        .take(top_k)
        .map(|(_, weight)| *weight)
        .sum();
    for (route_index, (expert_index, weight)) in ranked_scores.into_iter().take(top_k).enumerate() {
        indices[route_index] = expert_index;
        weights[route_index] = weight / selected_total;
    }
}

fn validate_route(
    scores: &[f32],
    shape: MoeRoutingShape,
    indices: &mut [usize],
    weights: &mut [f32],
) -> Result<(), MoeRoutingError> {
    validate_shape(shape)?;

    if scores.len() != shape.score_len() {
        return Err(MoeRoutingError::LengthMismatch {
            argument: "scores",
            expected: shape.score_len(),
            actual: scores.len(),
        });
    }

    validate_route_outputs(shape.route_len(), indices, weights)
}

fn validate_route_outputs(
    expected_len: usize,
    indices: &mut [usize],
    weights: &mut [f32],
) -> Result<(), MoeRoutingError> {
    if indices.len() != expected_len {
        return Err(MoeRoutingError::LengthMismatch {
            argument: "indices",
            expected: expected_len,
            actual: indices.len(),
        });
    }

    if weights.len() != expected_len {
        return Err(MoeRoutingError::LengthMismatch {
            argument: "weights",
            expected: expected_len,
            actual: weights.len(),
        });
    }

    Ok(())
}

fn validate_shape(shape: MoeRoutingShape) -> Result<(), MoeRoutingError> {
    if shape.expert_count == 0 || shape.top_k == 0 || shape.top_k > shape.expert_count {
        return Err(MoeRoutingError::InvalidShape(shape));
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoeRoutingError {
    InvalidShape(MoeRoutingShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    DeviceError(String),
}

impl From<TensorError> for MoeRoutingError {
    fn from(e: TensorError) -> Self {
        MoeRoutingError::DeviceError(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU MoE routing (sigmoid scoring with top-k selection).
pub async fn moe_route(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    let data = scores.to_host_async().await?;
    moe_route_host(&data, shape)
}

/// Async GPU MoE routing with softmax normalization.
pub async fn moe_route_softmax(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    let data = scores.to_host_async().await?;
    moe_route_softmax_host(&data, shape)
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

    /// Verifies that the MoE routing kernel reports HostFallback as its backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_moe_routing_kernels(),
            [MoeRoutingKernel {
                name: "moe_sigmoid_topk",
                backend: MoeRoutingBackend::HostFallback,
            }]
        );
    }

    /// Verifies sigmoid_host top-k routing selects the two highest-scored experts.
    ///
    /// This catches errors in the sigmoid_host transform or sort/select logic.
    #[test]
    fn routes_single_token_with_sigmoid_top_k() {
        let output = moe_route_token_host(&[0.0, 1.0, -1.0, 2.0], 2).unwrap();

        assert_eq!(output.indices, vec![3, 1]);
        approx_eq_slice(&output.weights, &[0.880797, 0.7310586]);
    }

    /// Verifies that multi-token routing processes each token's scores independently.
    ///
    /// This catches row-stride indexing errors in the batched routing path.
    #[test]
    fn routes_multiple_tokens_row_major() {
        let output = moe_route_host(
            &[0.0, 1.0, -1.0, 2.0, 5.0, -5.0, 0.5, 0.6],
            MoeRoutingShape::new(2, 4, 2),
        )
        .unwrap();

        assert_eq!(output.indices, vec![3, 1, 0, 3]);
        approx_eq_slice(
            &output.weights,
            &[0.880797, 0.7310586, 0.9933072, 0.6456563],
        );
    }

    /// Verifies that equal scores break ties toward the lower expert index.
    ///
    /// This catches unstable or reversed tie-breaking in the sort comparator.
    #[test]
    fn ties_break_toward_lower_expert_index() {
        let output = moe_route_token_host(&[0.0, 0.0, 1.0], 2).unwrap();

        assert_eq!(output.indices, vec![2, 0]);
        approx_eq(output.weights[0], 0.7310586);
        approx_eq(output.weights[1], 0.5);
    }

    /// Verifies that the _into variant writes into pre-allocated buffers.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn route_into_writes_existing_buffers() {
        let mut indices = [usize::MAX; 4];
        let mut weights = [-1.0; 4];

        moe_route_into_host(
            &[0.1, 0.9, 0.5, -1.0, 0.0, 3.0],
            MoeRoutingShape::new(2, 3, 2),
            &mut indices,
            &mut weights,
        )
        .unwrap();

        assert_eq!(indices, [1, 2, 2, 1]);
        approx_eq_slice(&weights, &[0.7109495, 0.62245935, 0.95257413, 0.5]);
    }

    /// Verifies softmax_host-normalized top-k routing with re-normalized selected weights.
    ///
    /// This catches errors in the softmax_host computation or top-k re-normalization.
    #[test]
    fn routes_with_softmax_normalized_top_k_weights() {
        let output = moe_route_softmax_host(
            &[
                0.4390189, 1.2967792, 2.4748528, 1.1023278, -1.263859, 0.51365805, 0.672, -0.11,
            ],
            MoeRoutingShape::new(1, 8, 2),
        )
        .unwrap();

        assert_eq!(output.indices, vec![2, 1]);
        approx_eq_slice(&output.weights, &[0.7646013, 0.23539874]);
    }

    /// Verifies that routing zero tokens produces empty output without error.
    ///
    /// This catches panics on zero-length inputs.
    #[test]
    fn empty_token_batch_produces_empty_routes() {
        let output = moe_route_host(&[], MoeRoutingShape::new(0, 4, 2)).unwrap();

        assert!(output.indices.is_empty());
        assert!(output.weights.is_empty());
    }

    /// Verifies that zero expert_count is rejected as an invalid shape.
    ///
    /// This catches missing dimension validation.
    #[test]
    fn rejects_invalid_shape() {
        let error = moe_route_host(&[], MoeRoutingShape::new(1, 0, 1)).unwrap_err();
        assert_eq!(
            error,
            MoeRoutingError::InvalidShape(MoeRoutingShape::new(1, 0, 1))
        );
    }

    /// Verifies that top_k > expert_count is rejected.
    ///
    /// This catches missing constraint validation between top_k and expert_count.
    #[test]
    fn rejects_top_k_larger_than_expert_count() {
        let error = moe_route_host(&[0.0, 1.0], MoeRoutingShape::new(1, 2, 3)).unwrap_err();
        assert_eq!(
            error,
            MoeRoutingError::InvalidShape(MoeRoutingShape::new(1, 2, 3))
        );
    }

    /// Verifies that a score buffer not matching token_count × expert_count is rejected.
    ///
    /// This catches missing score length validation.
    #[test]
    fn rejects_score_length_mismatch() {
        let error = moe_route_host(&[0.0, 1.0, 2.0], MoeRoutingShape::new(2, 2, 1)).unwrap_err();
        assert_eq!(
            error,
            MoeRoutingError::LengthMismatch {
                argument: "scores",
                expected: 4,
                actual: 3,
            }
        );
    }

    /// Verifies that a too-small indices buffer is rejected in the _into variant.
    ///
    /// This catches missing indices length validation.
    #[test]
    fn rejects_indices_length_mismatch() {
        let mut indices = [0; 1];
        let mut weights = [0.0; 2];
        let error = moe_route_into_host(
            &[0.0, 1.0],
            MoeRoutingShape::new(1, 2, 2),
            &mut indices,
            &mut weights,
        )
        .unwrap_err();

        assert_eq!(
            error,
            MoeRoutingError::LengthMismatch {
                argument: "indices",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that a too-small weights buffer is rejected in the _into variant.
    ///
    /// This catches missing weights length validation.
    #[test]
    fn rejects_weights_length_mismatch() {
        let mut indices = [0; 2];
        let mut weights = [0.0; 1];
        let error = moe_route_into_host(
            &[0.0, 1.0],
            MoeRoutingShape::new(1, 2, 2),
            &mut indices,
            &mut weights,
        )
        .unwrap_err();

        assert_eq!(
            error,
            MoeRoutingError::LengthMismatch {
                argument: "weights",
                expected: 2,
                actual: 1,
            }
        );
    }

    /// Verifies that top_k=0 is rejected in the single-token routing API.
    ///
    /// This catches missing validation in the convenience wrapper.
    #[test]
    fn rejects_single_token_invalid_top_k() {
        let error = moe_route_token_host(&[0.0, 1.0], 0).unwrap_err();
        assert_eq!(
            error,
            MoeRoutingError::InvalidShape(MoeRoutingShape::new(1, 2, 0))
        );
    }

    /// Verifies that the async GPU MoE routing matches the host fallback.
    /// This catches regressions in the GPU routing path.
    #[tokio::test]
    async fn gpu_moe_route_matches_host_fallback() {
        let scores = vec![0.1, 0.9, 0.4, 0.6];
        let shape = MoeRoutingShape::new(2, 2, 1);
        let expected = moe_route_host(&scores, shape).unwrap();
        let gpu_scores = GpuTensor::from_host(&scores, &[2, 2]).unwrap();
        let result = super::moe_route(&gpu_scores, shape).await.unwrap();
        assert_eq!(result.indices, expected.indices);
        assert_eq!(result.weights, expected.weights);
    }

    /// Verifies that the async GPU softmax MoE routing wrapper matches the host fallback when routing multiple tokens. This catches regressions in score transfer or top-k softmax normalization on the async wrapper path.
    #[tokio::test]
    async fn gpu_moe_route_softmax_matches_host_fallback() {
        let scores = vec![
            0.4390189, 1.2967792, 2.4748528, 1.1023278, -1.263859, 0.51365805,
        ];
        let shape = MoeRoutingShape::new(2, 3, 2);
        let expected = moe_route_softmax_host(&scores, shape).unwrap();
        let gpu_scores = GpuTensor::from_host(&scores, &[2, 3]).unwrap();

        let result = super::moe_route_softmax(&gpu_scores, shape).await.unwrap();

        assert_eq!(result.indices, expected.indices);
        approx_eq_slice(&result.weights, &expected.weights);
    }
}
