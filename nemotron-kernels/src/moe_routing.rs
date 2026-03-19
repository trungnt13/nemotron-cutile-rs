use crate::tensor::{GpuTensor, TensorError};
use crate::{activations::sigmoid_scalar, KernelStub};

pub const SPEC: KernelStub = KernelStub {
    name: "moe_routing",
    summary: "Top-k expert routing kernels with sigmoid_host scoring.",
};

#[cfg(any(target_os = "linux", test))]
const CUTILE_MAX_TOP_K: usize = 8;

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

#[cfg(target_os = "linux")]
const MOE_ROUTING_BACKEND: MoeRoutingBackend = MoeRoutingBackend::Cutile;
#[cfg(not(target_os = "linux"))]
const MOE_ROUTING_BACKEND: MoeRoutingBackend = MoeRoutingBackend::HostFallback;

pub const MOE_SIGMOID_TOPK: MoeRoutingKernel = MoeRoutingKernel {
    name: "moe_sigmoid_topk",
    backend: MOE_ROUTING_BACKEND,
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

fn validate_tensor_lengths(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<(), MoeRoutingError> {
    validate_shape(shape)?;

    if scores.numel() != shape.score_len() {
        return Err(MoeRoutingError::LengthMismatch {
            argument: "scores",
            expected: shape.score_len(),
            actual: scores.numel(),
        });
    }

    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn supports_cutile_moe_routing(shape: MoeRoutingShape) -> bool {
    shape.token_count > 0
        && matches!(shape.expert_count, 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128)
        && shape.top_k <= shape.expert_count
        && shape.top_k <= CUTILE_MAX_TOP_K
        && i32::try_from(shape.token_count).is_ok()
        && i32::try_from(shape.expert_count).is_ok()
        && i32::try_from(shape.top_k).is_ok()
}

fn backend_for_shape(_shape: MoeRoutingShape) -> MoeRoutingBackend {
    #[cfg(target_os = "linux")]
    {
        if supports_cutile_moe_routing(_shape) {
            return MoeRoutingBackend::Cutile;
        }
    }

    MoeRoutingBackend::HostFallback
}

#[cfg(target_os = "linux")]
mod cutile_impl {
    use super::{supports_cutile_moe_routing, MoeRoutingError, MoeRoutingOutput, MoeRoutingShape};
    use crate::tensor::GpuTensor;
    use cutile::tensor::ToHostVec;
    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel};

    #[cutile::module]
    mod cutile_moe_routing_kernel {
        use cutile::core::*;

        #[cutile::entry()]
        fn moe_sigmoid_topk<const EXPERT_COUNT: i32, const TOP_K: i32, const PADDED_TOP_K: i32>(
            indices: &mut Tensor<i32, { [1, PADDED_TOP_K] }>,
            weights: &mut Tensor<f32, { [1, PADDED_TOP_K] }>,
            scores: &Tensor<f32, { [-1, EXPERT_COUNT] }>,
            token_count: i32,
        ) {
            let token_index = get_tile_block_id().0;
            if token_index >= token_count {
                return;
            }
            let scores_part = scores.partition(const_shape![1, EXPERT_COUNT]);
            let score_tile: Tile<f32, { [1, EXPERT_COUNT] }> =
                scores_part.load([token_index, 0i32]);
            let score_tile: Tile<f32, { [EXPERT_COUNT] }> =
                score_tile.reshape(const_shape![EXPERT_COUNT]);

            let zero: Tile<f32, { [EXPERT_COUNT] }> = constant(0.0f32, const_shape![EXPERT_COUNT]);
            let one: Tile<f32, { [EXPERT_COUNT] }> = constant(1.0f32, const_shape![EXPERT_COUNT]);
            let nonnegative: Tile<bool, { [EXPERT_COUNT] }> = ge_tile(score_tile, zero);
            let positive_sigmoid: Tile<f32, { [EXPERT_COUNT] }> =
                one / (one + exp(zero - score_tile));
            let exp_score: Tile<f32, { [EXPERT_COUNT] }> = exp(score_tile);
            let negative_sigmoid: Tile<f32, { [EXPERT_COUNT] }> = exp_score / (one + exp_score);
            let mut remaining_weights: Tile<f32, { [EXPERT_COUNT] }> =
                select(nonnegative, positive_sigmoid, negative_sigmoid);

            let expert_positions: Tile<i32, { [EXPERT_COUNT] }> = iota(const_shape![EXPERT_COUNT]);
            let route_positions: Tile<i32, { [PADDED_TOP_K] }> = iota(const_shape![PADDED_TOP_K]);
            let mut selected_indices: Tile<i32, { [PADDED_TOP_K] }> =
                constant(0i32, const_shape![PADDED_TOP_K]);
            let mut selected_weights: Tile<f32, { [PADDED_TOP_K] }> =
                constant(0.0f32, const_shape![PADDED_TOP_K]);
            let neg_inf: Tile<f32, { [EXPERT_COUNT] }> =
                constant(f32::NEG_INFINITY, const_shape![EXPERT_COUNT]);
            let tie_break_sentinel: Tile<i32, { [EXPERT_COUNT] }> =
                broadcast_scalar(EXPERT_COUNT, const_shape![EXPERT_COUNT]);

            for route_index in 0i32..TOP_K {
                let max_weight_tile: Tile<f32, { [1] }> = reduce_max(remaining_weights, 0i32);
                let max_weight: f32 = tile_to_scalar(max_weight_tile.reshape(const_shape![]));
                let max_weight_broadcast: Tile<f32, { [EXPERT_COUNT] }> =
                    broadcast_scalar(max_weight, const_shape![EXPERT_COUNT]);
                let tied_max: Tile<bool, { [EXPERT_COUNT] }> =
                    eq_tile(remaining_weights, max_weight_broadcast);
                let candidate_indices: Tile<i32, { [EXPERT_COUNT] }> =
                    select(tied_max, expert_positions, tie_break_sentinel);
                let chosen_index_tile: Tile<i32, { [1] }> = reduce_min(candidate_indices, 0i32);
                let chosen_index: i32 = tile_to_scalar(chosen_index_tile.reshape(const_shape![]));

                let route_mask: Tile<bool, { [PADDED_TOP_K] }> = eq_tile(
                    route_positions,
                    broadcast_scalar(route_index, const_shape![PADDED_TOP_K]),
                );
                let chosen_indices_out: Tile<i32, { [PADDED_TOP_K] }> =
                    broadcast_scalar(chosen_index, const_shape![PADDED_TOP_K]);
                let chosen_weights_out: Tile<f32, { [PADDED_TOP_K] }> =
                    broadcast_scalar(max_weight, const_shape![PADDED_TOP_K]);
                selected_indices = select(route_mask, chosen_indices_out, selected_indices);
                selected_weights = select(route_mask, chosen_weights_out, selected_weights);

                let chosen_mask: Tile<bool, { [EXPERT_COUNT] }> = eq_tile(
                    expert_positions,
                    broadcast_scalar(chosen_index, const_shape![EXPERT_COUNT]),
                );
                remaining_weights = select(chosen_mask, neg_inf, remaining_weights);
            }

            indices.store(selected_indices.reshape(const_shape![1, PADDED_TOP_K]));
            weights.store(selected_weights.reshape(const_shape![1, PADDED_TOP_K]));
        }
    }

    use cutile_moe_routing_kernel::moe_sigmoid_topk_async;

    fn device_error(prefix: &str, error: impl std::fmt::Debug) -> MoeRoutingError {
        MoeRoutingError::DeviceError(format!("{prefix}: {error:?}"))
    }

    pub(super) async fn moe_route(
        scores: &GpuTensor,
        shape: MoeRoutingShape,
    ) -> Result<Option<MoeRoutingOutput>, MoeRoutingError> {
        if !supports_cutile_moe_routing(shape) {
            return Ok(None);
        }

        let token_count_i32 = i32::try_from(shape.token_count).map_err(|_| {
            MoeRoutingError::DeviceError(format!(
                "cutile MoE token_count {} exceeds the supported i32 launch bound",
                shape.token_count
            ))
        })?;
        let padded_top_k = shape.top_k.next_power_of_two();
        let padded_token_count = shape.token_count.next_power_of_two();
        let padded_top_k_i32 = i32::try_from(padded_top_k).map_err(|_| {
            MoeRoutingError::DeviceError(format!(
                "cutile MoE padded top_k {padded_top_k} exceeds the supported i32 launch bound"
            ))
        })?;
        let scores_tensor = scores
            .cutile_tensor_for_shape(&[shape.token_count, shape.expert_count])
            .await?;
        let output_indices = cutile::api::zeros::<2, i32>([padded_token_count, padded_top_k])
            .partition([1, padded_top_k_i32]);
        let output_weights = cutile::api::zeros::<2, f32>([padded_token_count, padded_top_k])
            .partition([1, padded_top_k_i32]);
        let generics = vec![
            shape.expert_count.to_string(),
            shape.top_k.to_string(),
            padded_top_k.to_string(),
        ];
        let (output_indices, output_weights, _scores, _token_count) = moe_sigmoid_topk_async(
            output_indices,
            output_weights,
            scores_tensor.device_operation(),
            cutile::tile_kernel::value(token_count_i32),
        )
        .generics(generics)
        .await
        .map_err(|error| device_error("cutile MoE launch failed", error))?;
        let output_indices = output_indices.unpartition();
        let output_weights = output_weights.unpartition();
        let indices = output_indices
            .to_host_vec()
            .await
            .map_err(|error| device_error("cutile MoE indices transfer failed", error))?;
        let weights = output_weights
            .to_host_vec()
            .await
            .map_err(|error| device_error("cutile MoE weights transfer failed", error))?;
        let mut compact_indices = Vec::with_capacity(shape.route_len());
        let mut compact_weights = Vec::with_capacity(shape.route_len());
        for token_index in 0..shape.token_count {
            let row_start = token_index * padded_top_k;
            let row_end = row_start + shape.top_k;
            compact_indices.extend(
                indices[row_start..row_end]
                    .iter()
                    .copied()
                    .map(|index| {
                        usize::try_from(index).map_err(|_| {
                            MoeRoutingError::DeviceError(format!(
                                "cutile MoE index {index} could not convert to usize"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            compact_weights.extend_from_slice(&weights[row_start..row_end]);
        }

        Ok(Some(MoeRoutingOutput {
            indices: compact_indices,
            weights: compact_weights,
        }))
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

async fn moe_route_host_bridge(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    let data = scores.to_host_async().await?;
    moe_route_host(&data, shape)
}

async fn moe_route_softmax_host_bridge(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    let data = scores.to_host_async().await?;
    moe_route_softmax_host(&data, shape)
}

/// Async GPU MoE routing (sigmoid scoring with top-k selection).
pub async fn moe_route(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    validate_tensor_lengths(scores, shape)?;

    match backend_for_shape(shape) {
        MoeRoutingBackend::Cutile => {
            #[cfg(target_os = "linux")]
            {
                return cutile_impl::moe_route(scores, shape).await?.ok_or_else(|| {
                    MoeRoutingError::DeviceError(
                        "cutile MoE routing reported support but did not launch".to_string(),
                    )
                });
            }

            #[cfg(not(target_os = "linux"))]
            {
                unreachable!("non-Linux platforms never select the cutile MoE routing backend")
            }
        }
        MoeRoutingBackend::HostFallback => moe_route_host_bridge(scores, shape).await,
    }
}

/// Async GPU MoE routing with softmax normalization.
pub async fn moe_route_softmax(
    scores: &GpuTensor,
    shape: MoeRoutingShape,
) -> Result<MoeRoutingOutput, MoeRoutingError> {
    validate_tensor_lengths(scores, shape)?;
    moe_route_softmax_host_bridge(scores, shape).await
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(lhs: f32, rhs: f32) {
        approx_eq_with_tolerance(lhs, rhs, 1e-6);
    }

    fn approx_eq_with_tolerance(lhs: f32, rhs: f32, tolerance: f32) {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= tolerance,
            "values differ: left={lhs:?}, right={rhs:?}, diff={diff:?}, tolerance={tolerance:?}"
        );
    }

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32]) {
        approx_eq_slice_with_tolerance(lhs, rhs, 1e-6);
    }

    fn approx_eq_slice_with_tolerance(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths differ");
        for (index, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "index {index}: left={left:?}, right={right:?}, diff={diff:?}, tolerance={tolerance:?}"
            );
        }
    }

    /// Verifies that the MoE routing kernel registry reports the current platform's primary backend.
    ///
    /// This catches accidental backend metadata regressions as Linux enables cutile routing while other platforms keep host fallback.
    #[test]
    fn reports_platform_primary_backend() {
        #[cfg(target_os = "linux")]
        let expected = [MoeRoutingKernel {
            name: "moe_sigmoid_topk",
            backend: MoeRoutingBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [MoeRoutingKernel {
            name: "moe_sigmoid_topk",
            backend: MoeRoutingBackend::HostFallback,
        }];

        assert_eq!(supported_moe_routing_kernels(), expected);
    }

    /// Verifies that the cutile routing heuristic only accepts the planned expert-count and top-k envelope.
    ///
    /// This catches regressions where Linux would try to launch the cutile kernel for unsupported MoE shapes instead of falling back cleanly.
    #[test]
    fn cutile_support_heuristic_respects_shape_boundaries() {
        assert!(supports_cutile_moe_routing(MoeRoutingShape::new(2, 8, 2)));
        assert!(supports_cutile_moe_routing(MoeRoutingShape::new(1, 128, 6)));
        assert!(!supports_cutile_moe_routing(MoeRoutingShape::new(0, 8, 2)));
        assert!(!supports_cutile_moe_routing(MoeRoutingShape::new(2, 5, 2)));
        assert!(!supports_cutile_moe_routing(MoeRoutingShape::new(2, 8, 9)));
    }

    /// Verifies sigmoid_host top-k routing selects the two highest-scored experts when routing one token.
    ///
    /// This catches errors in the sigmoid_host transform or sort/select logic.
    #[test]
    fn routes_single_token_with_sigmoid_top_k() {
        let output = moe_route_token_host(&[0.0, 1.0, -1.0, 2.0], 2).unwrap();

        assert_eq!(output.indices, vec![3, 1]);
        approx_eq_slice(&output.weights, &[0.880797, 0.7310586]);
    }

    /// Verifies that multi-token routing processes each token's scores independently when the input is row-major.
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

    /// Verifies that equal scores break ties toward the lower expert index when routing a token.
    ///
    /// This catches unstable or reversed tie-breaking in the sort comparator.
    #[test]
    fn ties_break_toward_lower_expert_index() {
        let output = moe_route_token_host(&[0.0, 0.0, 1.0], 2).unwrap();

        assert_eq!(output.indices, vec![2, 0]);
        approx_eq(output.weights[0], 0.7310586);
        approx_eq(output.weights[1], 0.5);
    }

    /// Verifies that the _into variant writes routing results into caller-provided buffers when batching tokens.
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

    /// Verifies softmax_host-normalized top-k routing re-normalizes only the selected weights when softmax routing is requested.
    ///
    /// This catches errors in the softmax_host computation or selected-weight normalization contract.
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

    /// Verifies that routing zero tokens produces empty output when expert and top-k dimensions are still valid.
    ///
    /// This catches panics on zero-length batches.
    #[test]
    fn empty_token_batch_produces_empty_routes() {
        let output = moe_route_host(&[], MoeRoutingShape::new(0, 4, 2)).unwrap();

        assert!(output.indices.is_empty());
        assert!(output.weights.is_empty());
    }

    /// Verifies that zero expert_count is rejected when validating the routing shape.
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

    /// Verifies that top_k greater than expert_count is rejected when validating a routing request.
    ///
    /// This catches missing top-k versus expert-count validation.
    #[test]
    fn rejects_top_k_larger_than_expert_count() {
        let error = moe_route_host(&[0.0, 1.0], MoeRoutingShape::new(1, 2, 3)).unwrap_err();
        assert_eq!(
            error,
            MoeRoutingError::InvalidShape(MoeRoutingShape::new(1, 2, 3))
        );
    }

    /// Verifies that a score buffer not matching token_count × expert_count is rejected when routing on host.
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

    /// Verifies that a too-small indices buffer is rejected when using the _into host API.
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

    /// Verifies that a too-small weights buffer is rejected when using the _into host API.
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

    /// Verifies that top_k=0 is rejected when routing through the single-token convenience API.
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

    /// Verifies that the async GPU sigmoid-routing API matches the host reference when given a cutile-supported tiny shape.
    ///
    /// This catches regressions in GPU routing parity while still exercising Linux cutile dispatch where available.
    #[tokio::test]
    async fn gpu_moe_route_matches_host_reference() {
        let scores = vec![0.1, 0.9, 0.4, 0.6];
        let shape = MoeRoutingShape::new(2, 2, 1);
        let expected = moe_route_host(&scores, shape).unwrap();
        let gpu_scores = GpuTensor::from_host(&scores, &[2, 2]).unwrap();
        let result = super::moe_route(&gpu_scores, shape).await.unwrap();

        #[cfg(target_os = "linux")]
        assert_eq!(backend_for_shape(shape), MoeRoutingBackend::Cutile);

        assert_eq!(result.indices, expected.indices);
        approx_eq_slice_with_tolerance(&result.weights, &expected.weights, 5e-6);
    }

    /// Verifies that the async GPU softmax-routing API matches the host reference when routing multiple tokens.
    ///
    /// This catches regressions in score transfer or top-k softmax normalization on the async wrapper path.
    #[tokio::test]
    async fn gpu_moe_route_softmax_matches_host_reference() {
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

    /// Verifies that the Linux GPU routing path matches the host reference when exercising a model-like 128-expert, top-6 shape.
    ///
    /// This catches regressions in repeated top-k selection, stable sigmoid scoring, or tie-breaking on the real cutile MoE kernel.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn gpu_moe_route_uses_cutile_for_model_like_shape() {
        let shape = MoeRoutingShape::new(2, 128, 6);
        assert_eq!(backend_for_shape(shape), MoeRoutingBackend::Cutile);

        let scores = (0..shape.score_len())
            .map(|index| {
                let token = index / shape.expert_count;
                let expert = index % shape.expert_count;
                ((expert % 17) as f32 - 8.0) * 0.375 + token as f32 * 0.125 - 0.05
            })
            .collect::<Vec<_>>();
        let expected = moe_route_host(&scores, shape).unwrap();
        let gpu_scores =
            GpuTensor::from_host(&scores, &[shape.token_count, shape.expert_count]).unwrap();
        let result = super::moe_route(&gpu_scores, shape).await.unwrap();

        assert_eq!(result.indices, expected.indices);
        approx_eq_slice_with_tolerance(&result.weights, &expected.weights, 5e-6);
    }
}
