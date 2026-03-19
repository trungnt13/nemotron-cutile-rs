use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "attention",
    summary: "Grouped-query attention kernels.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionKernel {
    pub name: &'static str,
    pub backend: AttentionBackend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionShape {
    pub batch_size: usize,
    pub query_sequence_len: usize,
    pub key_value_sequence_len: usize,
    pub query_head_count: usize,
    pub key_value_head_count: usize,
    pub head_dim: usize,
}

impl AttentionShape {
    pub const fn new(
        batch_size: usize,
        query_sequence_len: usize,
        key_value_sequence_len: usize,
        query_head_count: usize,
        key_value_head_count: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            batch_size,
            query_sequence_len,
            key_value_sequence_len,
            query_head_count,
            key_value_head_count,
            head_dim,
        }
    }

    pub const fn query_len(self) -> usize {
        self.batch_size * self.query_sequence_len * self.query_head_count * self.head_dim
    }

    pub const fn key_len(self) -> usize {
        self.batch_size * self.key_value_sequence_len * self.key_value_head_count * self.head_dim
    }

    pub const fn value_len(self) -> usize {
        self.key_len()
    }

    pub const fn output_len(self) -> usize {
        self.query_len()
    }

    pub const fn score_len(self) -> usize {
        self.batch_size
            * self.query_sequence_len
            * self.query_head_count
            * self.key_value_sequence_len
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOptions {
    pub causal: bool,
    pub query_position_offset: usize,
    pub softmax_scale: Option<f32>,
}

impl AttentionOptions {
    pub fn resolve_scale(self, head_dim: usize) -> Result<f32, AttentionError> {
        let scale = self
            .softmax_scale
            .unwrap_or_else(|| (head_dim as f32).sqrt().recip());

        if !scale.is_finite() || scale <= 0.0 {
            return Err(AttentionError::InvalidScale(scale));
        }

        Ok(scale)
    }
}

impl Default for AttentionOptions {
    fn default() -> Self {
        Self {
            causal: false,
            query_position_offset: 0,
            softmax_scale: None,
        }
    }
}

#[cfg(target_os = "linux")]
const ATTENTION_BACKEND: AttentionBackend = AttentionBackend::Cutile;
#[cfg(not(target_os = "linux"))]
const ATTENTION_BACKEND: AttentionBackend = AttentionBackend::HostFallback;

pub const GROUPED_QUERY_ATTENTION: AttentionKernel = AttentionKernel {
    name: "grouped_query_attention",
    backend: ATTENTION_BACKEND,
};

pub fn supported_attention_kernels() -> [AttentionKernel; 1] {
    [GROUPED_QUERY_ATTENTION]
}

#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const CUTILE_ATTENTION_SCORE_TILE_WIDTH: usize = 16;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const CUTILE_ATTENTION_DEPTH_TILE: usize = 8;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const CUTILE_ATTENTION_VALUE_TILE: usize = 8;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const CUTILE_ATTENTION_SOFTMAX_MAX_WIDTH: usize = 4096;

pub fn attention_scores_host(
    query: &[f32],
    key: &[f32],
    shape: AttentionShape,
    options: AttentionOptions,
) -> Result<Vec<f32>, AttentionError> {
    validate_shape(query, key, None, shape)?;
    let scale = options.resolve_scale(shape.head_dim)?;
    let kv_group_size = shape.query_head_count / shape.key_value_head_count;
    let mut scores = vec![f32::NEG_INFINITY; shape.score_len()];

    for batch in 0..shape.batch_size {
        for query_index in 0..shape.query_sequence_len {
            let max_key_index = if options.causal {
                (options.query_position_offset + query_index).min(shape.key_value_sequence_len - 1)
            } else {
                shape.key_value_sequence_len - 1
            };

            for query_head in 0..shape.query_head_count {
                let key_head = query_head / kv_group_size;
                let row_offset = score_offset(batch, query_index, query_head, shape);

                for key_index in 0..shape.key_value_sequence_len {
                    if options.causal && key_index > max_key_index {
                        continue;
                    }

                    let mut score = 0.0_f64;
                    for dim in 0..shape.head_dim {
                        let query_value = query[tensor_offset(
                            batch,
                            query_index,
                            query_head,
                            dim,
                            shape.query_sequence_len,
                            shape.query_head_count,
                            shape.head_dim,
                        )];
                        let key_value = key[tensor_offset(
                            batch,
                            key_index,
                            key_head,
                            dim,
                            shape.key_value_sequence_len,
                            shape.key_value_head_count,
                            shape.head_dim,
                        )];
                        score += f64::from(query_value) * f64::from(key_value);
                    }

                    scores[row_offset + key_index] = (score as f32) * scale;
                }
            }
        }
    }

    Ok(scores)
}

pub fn scaled_dot_product_attention_host(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: AttentionShape,
    options: AttentionOptions,
) -> Result<Vec<f32>, AttentionError> {
    let mut output = vec![0.0; shape.output_len()];
    scaled_dot_product_attention_into_host(query, key, value, shape, options, &mut output)?;
    Ok(output)
}

pub fn scaled_dot_product_attention_into_host(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: AttentionShape,
    options: AttentionOptions,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    validate_shape(query, key, Some(value), shape)?;

    if output.len() != shape.output_len() {
        return Err(AttentionError::LengthMismatch {
            argument: "output",
            expected: shape.output_len(),
            actual: output.len(),
        });
    }

    let scores = attention_scores_host(query, key, shape, options)?;
    let kv_group_size = shape.query_head_count / shape.key_value_head_count;
    let mut weights = vec![0.0_f32; shape.key_value_sequence_len];

    for batch in 0..shape.batch_size {
        for query_index in 0..shape.query_sequence_len {
            for query_head in 0..shape.query_head_count {
                let score_row_offset = score_offset(batch, query_index, query_head, shape);
                let score_row =
                    &scores[score_row_offset..score_row_offset + shape.key_value_sequence_len];

                let mut max_score = f32::NEG_INFINITY;
                for score in score_row.iter().copied().filter(|score| score.is_finite()) {
                    max_score = max_score.max(score);
                }

                if !max_score.is_finite() {
                    return Err(AttentionError::NoValidAttentionTargets {
                        batch,
                        query_index,
                        query_head,
                    });
                }

                let mut partition = 0.0_f64;
                for (weight, score) in weights.iter_mut().zip(score_row.iter().copied()) {
                    if score.is_finite() {
                        let unnormalized = (score - max_score).exp();
                        *weight = unnormalized;
                        partition += f64::from(unnormalized);
                    } else {
                        *weight = 0.0;
                    }
                }

                if partition == 0.0 {
                    return Err(AttentionError::ZeroPartition {
                        batch,
                        query_index,
                        query_head,
                    });
                }

                let inv_partition = (1.0 / partition) as f32;
                let value_head = query_head / kv_group_size;
                for dim in 0..shape.head_dim {
                    let mut acc = 0.0_f64;
                    for (key_index, weight) in weights.iter().copied().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }

                        let value_index = tensor_offset(
                            batch,
                            key_index,
                            value_head,
                            dim,
                            shape.key_value_sequence_len,
                            shape.key_value_head_count,
                            shape.head_dim,
                        );
                        acc += f64::from(weight * inv_partition) * f64::from(value[value_index]);
                    }

                    let output_index = tensor_offset(
                        batch,
                        query_index,
                        query_head,
                        dim,
                        shape.query_sequence_len,
                        shape.query_head_count,
                        shape.head_dim,
                    );
                    output[output_index] = acc as f32;
                }
            }
        }
    }

    Ok(())
}

fn validate_shape(
    query: &[f32],
    key: &[f32],
    value: Option<&[f32]>,
    shape: AttentionShape,
) -> Result<(), AttentionError> {
    if shape.batch_size == 0
        || shape.query_sequence_len == 0
        || shape.key_value_sequence_len == 0
        || shape.query_head_count == 0
        || shape.key_value_head_count == 0
        || shape.head_dim == 0
    {
        return Err(AttentionError::InvalidShape(shape));
    }

    if !shape
        .query_head_count
        .is_multiple_of(shape.key_value_head_count)
    {
        return Err(AttentionError::HeadCountMismatch {
            query_head_count: shape.query_head_count,
            key_value_head_count: shape.key_value_head_count,
        });
    }

    if query.len() != shape.query_len() {
        return Err(AttentionError::LengthMismatch {
            argument: "query",
            expected: shape.query_len(),
            actual: query.len(),
        });
    }

    if key.len() != shape.key_len() {
        return Err(AttentionError::LengthMismatch {
            argument: "key",
            expected: shape.key_len(),
            actual: key.len(),
        });
    }

    if let Some(value) = value {
        if value.len() != shape.value_len() {
            return Err(AttentionError::LengthMismatch {
                argument: "value",
                expected: shape.value_len(),
                actual: value.len(),
            });
        }
    }

    Ok(())
}

fn validate_tensor_lengths(
    query: &GpuTensor,
    key: &GpuTensor,
    value: &GpuTensor,
    shape: AttentionShape,
) -> Result<(), AttentionError> {
    if shape.batch_size == 0
        || shape.query_sequence_len == 0
        || shape.key_value_sequence_len == 0
        || shape.query_head_count == 0
        || shape.key_value_head_count == 0
        || shape.head_dim == 0
    {
        return Err(AttentionError::InvalidShape(shape));
    }

    if !shape
        .query_head_count
        .is_multiple_of(shape.key_value_head_count)
    {
        return Err(AttentionError::HeadCountMismatch {
            query_head_count: shape.query_head_count,
            key_value_head_count: shape.key_value_head_count,
        });
    }

    if query.numel() != shape.query_len() {
        return Err(AttentionError::LengthMismatch {
            argument: "query",
            expected: shape.query_len(),
            actual: query.numel(),
        });
    }

    if key.numel() != shape.key_len() {
        return Err(AttentionError::LengthMismatch {
            argument: "key",
            expected: shape.key_len(),
            actual: key.numel(),
        });
    }

    if value.numel() != shape.value_len() {
        return Err(AttentionError::LengthMismatch {
            argument: "value",
            expected: shape.value_len(),
            actual: value.numel(),
        });
    }

    Ok(())
}

#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn supports_cutile_attention(shape: AttentionShape, options: AttentionOptions) -> bool {
    let Some(rows) = shape
        .batch_size
        .checked_mul(shape.query_sequence_len)
        .and_then(|rows| rows.checked_mul(shape.query_head_count))
    else {
        return false;
    };

    rows > 0
        && i32::try_from(rows).is_ok()
        && i32::try_from(shape.batch_size).is_ok()
        && i32::try_from(shape.query_sequence_len).is_ok()
        && i32::try_from(shape.key_value_sequence_len).is_ok()
        && i32::try_from(shape.query_head_count).is_ok()
        && i32::try_from(shape.key_value_head_count).is_ok()
        && i32::try_from(shape.head_dim).is_ok()
        && i32::try_from(options.query_position_offset).is_ok()
        && shape.key_value_sequence_len <= CUTILE_ATTENTION_SOFTMAX_MAX_WIDTH
        && shape.key_value_sequence_len.is_power_of_two()
        && shape.key_value_sequence_len % CUTILE_ATTENTION_SCORE_TILE_WIDTH == 0
        && shape.head_dim % CUTILE_ATTENTION_DEPTH_TILE == 0
        && shape.key_value_sequence_len % CUTILE_ATTENTION_VALUE_TILE == 0
}

fn backend_for_shape(_shape: AttentionShape, _options: AttentionOptions) -> AttentionBackend {
    #[cfg(target_os = "linux")]
    {
        if supports_cutile_attention(_shape, _options) {
            return AttentionBackend::Cutile;
        }
    }

    AttentionBackend::HostFallback
}

fn tensor_offset(
    batch: usize,
    sequence_index: usize,
    head_index: usize,
    dim: usize,
    sequence_len: usize,
    head_count: usize,
    head_dim: usize,
) -> usize {
    (((batch * sequence_len + sequence_index) * head_count + head_index) * head_dim) + dim
}

fn score_offset(
    batch: usize,
    query_index: usize,
    query_head: usize,
    shape: AttentionShape,
) -> usize {
    ((batch * shape.query_sequence_len + query_index) * shape.query_head_count + query_head)
        * shape.key_value_sequence_len
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttentionError {
    InvalidShape(AttentionShape),
    HeadCountMismatch {
        query_head_count: usize,
        key_value_head_count: usize,
    },
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidScale(f32),
    NoValidAttentionTargets {
        batch: usize,
        query_index: usize,
        query_head: usize,
    },
    ZeroPartition {
        batch: usize,
        query_index: usize,
        query_head: usize,
    },
    DeviceError(String),
}

impl From<TensorError> for AttentionError {
    fn from(e: TensorError) -> Self {
        AttentionError::DeviceError(e.to_string())
    }
}

#[cfg(target_os = "linux")]
#[cutile::module]
mod cutile_attention_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn attention_scores<
        const BN: i32,
        const BD: i32,
        const QUERY_SEQUENCE_LEN: i32,
        const QUERY_HEAD_COUNT: i32,
        const KEY_VALUE_HEAD_COUNT: i32,
        const HEAD_DIM: i32,
    >(
        query: &Tensor<f32, { [-1, -1] }>,
        key: &Tensor<f32, { [-1, -1, -1, -1] }>,
        output: &mut Tensor<f32, { [1, BN] }>,
        scale: f32,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let key_tile = pid.1;
        let rows_per_batch = QUERY_SEQUENCE_LEN * QUERY_HEAD_COUNT;
        let batch = row / rows_per_batch;
        let row_in_batch = row % rows_per_batch;
        let query_head = row_in_batch % QUERY_HEAD_COUNT;
        let kv_group_size = QUERY_HEAD_COUNT / KEY_VALUE_HEAD_COUNT;
        let key_head = query_head / kv_group_size;
        let query_partition: Partition<f32, { [1, BD] }> = query.partition(const_shape![1, BD]);
        let key_partition: Partition<f32, { [1, 1, BD, BN] }> =
            key.partition_permuted(const_shape![1, 1, BD, BN], const_array![0, 2, 3, 1]);
        let mut scores: Tile<f32, { [1, BN] }> = broadcast_scalar(0.0f32, output.shape());

        for depth_tile in 0i32..(HEAD_DIM / BD) {
            let query_tile: Tile<f32, { [1, BD] }> = query_partition.load([row, depth_tile]);
            let query_tile: Tile<f32, { [BD, 1] }> = query_tile.reshape(const_shape![BD, 1]);
            let key_tile_values: Tile<f32, { [1, 1, BD, BN] }> =
                key_partition.load([batch, key_head, depth_tile, key_tile]);
            let key_tile_values: Tile<f32, { [BD, BN] }> =
                key_tile_values.reshape(const_shape![BD, BN]);
            let query_tile: Tile<f32, { [BD, BN] }> = query_tile.broadcast(const_shape![BD, BN]);
            let product: Tile<f32, { [BD, BN] }> = query_tile * key_tile_values;
            let partial: Tile<f32, { [BN] }> = reduce_sum(product, 0i32);
            let partial: Tile<f32, { [1, BN] }> = partial.reshape(output.shape());
            scores = scores + partial;
        }

        let scaled_scores: Tile<f32, { [1, BN] }> =
            scores * broadcast_scalar(scale, output.shape());
        output.store(scaled_scores);
    }

    #[cutile::entry()]
    fn row_softmax<const BN: i32>(
        input: &Tensor<f32, { [-1, -1] }>,
        output: &mut Tensor<f32, { [1, BN] }>,
    ) {
        let tile_input: Tile<f32, { [1, BN] }> = load_tile_like_2d(input, output);
        let tile_max: Tile<f32, { [1] }> = reduce_max(tile_input, 1i32);
        let tile_max: Tile<f32, { [1, BN] }> = tile_max
            .reshape(const_shape![1, 1])
            .broadcast(output.shape());
        let numerator: Tile<f32, { [1, BN] }> = exp(tile_input - tile_max);
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerator, 1i32);
        let denominator: Tile<f32, { [1, BN] }> = denominator
            .reshape(const_shape![1, 1])
            .broadcast(output.shape());
        output.store(numerator / denominator);
    }

    #[cutile::entry()]
    fn masked_row_softmax<
        const BN: i32,
        const QUERY_SEQUENCE_LEN: i32,
        const QUERY_HEAD_COUNT: i32,
    >(
        input: &Tensor<f32, { [-1, -1] }>,
        output: &mut Tensor<f32, { [1, BN] }>,
        query_position_offset: i32,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let row_in_batch = row % (QUERY_SEQUENCE_LEN * QUERY_HEAD_COUNT);
        let query_index = row_in_batch / QUERY_HEAD_COUNT;
        let mut max_key_index = query_position_offset + query_index;
        if max_key_index >= BN {
            max_key_index = BN - 1i32;
        }

        let tile_input: Tile<f32, { [1, BN] }> = load_tile_like_2d(input, output);
        let key_indices: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let valid_keys: Tile<bool, { [BN] }> = le_tile(
            key_indices,
            broadcast_scalar(max_key_index, const_shape![BN]),
        );
        let negative_inf = -3.4028235e38f32;
        let masked_input: Tile<f32, { [BN] }> = select(
            valid_keys,
            tile_input.reshape(const_shape![BN]),
            broadcast_scalar(negative_inf, const_shape![BN]),
        );
        let masked_input: Tile<f32, { [1, BN] }> = masked_input.reshape(output.shape());
        let tile_max: Tile<f32, { [1] }> = reduce_max(masked_input, 1i32);
        let tile_max: Tile<f32, { [1, BN] }> = tile_max
            .reshape(const_shape![1, 1])
            .broadcast(output.shape());
        let numerator: Tile<f32, { [1, BN] }> = exp(masked_input - tile_max);
        let denominator: Tile<f32, { [1] }> = reduce_sum(numerator, 1i32);
        let denominator: Tile<f32, { [1, BN] }> = denominator
            .reshape(const_shape![1, 1])
            .broadcast(output.shape());
        output.store(numerator / denominator);
    }

    #[cutile::entry()]
    fn attention_weighted_values<
        const BK: i32,
        const BD: i32,
        const QUERY_SEQUENCE_LEN: i32,
        const QUERY_HEAD_COUNT: i32,
        const KEY_VALUE_HEAD_COUNT: i32,
        const KEY_SEQUENCE_LEN: i32,
    >(
        weights: &Tensor<f32, { [-1, -1] }>,
        value: &Tensor<f32, { [-1, -1, -1, -1] }>,
        output: &mut Tensor<f32, { [1, BD] }>,
    ) {
        let pid = get_tile_block_id();
        let row = pid.0;
        let dim_tile = pid.1;
        let rows_per_batch = QUERY_SEQUENCE_LEN * QUERY_HEAD_COUNT;
        let batch = row / rows_per_batch;
        let row_in_batch = row % rows_per_batch;
        let query_head = row_in_batch % QUERY_HEAD_COUNT;
        let kv_group_size = QUERY_HEAD_COUNT / KEY_VALUE_HEAD_COUNT;
        let value_head = query_head / kv_group_size;
        let weight_partition: Partition<f32, { [1, BK] }> = weights.partition(const_shape![1, BK]);
        let value_partition: Partition<f32, { [1, 1, BK, BD] }> =
            value.partition_permuted(const_shape![1, 1, BK, BD], const_array![0, 2, 1, 3]);
        let mut acc: Tile<f32, { [1, BD] }> = broadcast_scalar(0.0f32, output.shape());

        for key_tile in 0i32..(KEY_SEQUENCE_LEN / BK) {
            let weight_tile: Tile<f32, { [1, BK] }> = weight_partition.load([row, key_tile]);
            let weight_tile: Tile<f32, { [BK, 1] }> = weight_tile.reshape(const_shape![BK, 1]);
            let weight_tile: Tile<f32, { [BK, BD] }> = weight_tile.broadcast(const_shape![BK, BD]);
            let value_tile: Tile<f32, { [1, 1, BK, BD] }> =
                value_partition.load([batch, value_head, key_tile, dim_tile]);
            let value_tile: Tile<f32, { [BK, BD] }> = value_tile.reshape(const_shape![BK, BD]);
            let product: Tile<f32, { [BK, BD] }> = weight_tile * value_tile;
            let partial: Tile<f32, { [BD] }> = reduce_sum(product, 0i32);
            let partial: Tile<f32, { [1, BD] }> = partial.reshape(output.shape());
            acc = acc + partial;
        }

        output.store(acc);
    }
}

#[cfg(target_os = "linux")]
async fn scaled_dot_product_attention_cutile(
    query: &GpuTensor,
    key: &GpuTensor,
    value: &GpuTensor,
    shape: AttentionShape,
    options: AttentionOptions,
    scale: f32,
) -> Result<Option<GpuTensor>, AttentionError> {
    use std::sync::Arc;

    use cutile::tile_kernel::{IntoDeviceOperation, IntoDeviceOperationPartition, TileKernel};

    if !supports_cutile_attention(shape, options) {
        return Ok(None);
    }

    let rows = shape.batch_size * shape.query_sequence_len * shape.query_head_count;
    let _rows_i32 = i32::try_from(rows).map_err(|_| {
        AttentionError::DeviceError("attention row count overflowed i32".to_string())
    })?;
    let score_tile_width_i32 = i32::try_from(CUTILE_ATTENTION_SCORE_TILE_WIDTH).map_err(|_| {
        AttentionError::DeviceError("attention score tile width overflowed i32".to_string())
    })?;
    let depth_tile_i32 = i32::try_from(CUTILE_ATTENTION_DEPTH_TILE).map_err(|_| {
        AttentionError::DeviceError("attention depth tile overflowed i32".to_string())
    })?;
    let value_tile_i32 = i32::try_from(CUTILE_ATTENTION_VALUE_TILE).map_err(|_| {
        AttentionError::DeviceError("attention value tile overflowed i32".to_string())
    })?;
    let key_sequence_len_i32 = i32::try_from(shape.key_value_sequence_len).map_err(|_| {
        AttentionError::DeviceError(
            "attention key/value sequence length overflowed i32".to_string(),
        )
    })?;
    let query_sequence_len_i32 = i32::try_from(shape.query_sequence_len).map_err(|_| {
        AttentionError::DeviceError("attention query sequence length overflowed i32".to_string())
    })?;
    let query_head_count_i32 = i32::try_from(shape.query_head_count).map_err(|_| {
        AttentionError::DeviceError("attention query head count overflowed i32".to_string())
    })?;
    let key_value_head_count_i32 = i32::try_from(shape.key_value_head_count).map_err(|_| {
        AttentionError::DeviceError("attention key/value head count overflowed i32".to_string())
    })?;
    let head_dim_i32 = i32::try_from(shape.head_dim).map_err(|_| {
        AttentionError::DeviceError("attention head dimension overflowed i32".to_string())
    })?;
    let query_position_offset_i32 = i32::try_from(options.query_position_offset).map_err(|_| {
        AttentionError::DeviceError("attention query position offset overflowed i32".to_string())
    })?;

    let query_tensor = query
        .cutile_tensor_for_shape(&[rows, shape.head_dim])
        .await?;
    let key_tensor = key
        .cutile_tensor_for_shape(&[
            shape.batch_size,
            shape.key_value_sequence_len,
            shape.key_value_head_count,
            shape.head_dim,
        ])
        .await?;
    let value_tensor = value
        .cutile_tensor_for_shape(&[
            shape.batch_size,
            shape.key_value_sequence_len,
            shape.key_value_head_count,
            shape.head_dim,
        ])
        .await?;

    let score_output = cutile::api::zeros::<2, f32>([rows, shape.key_value_sequence_len])
        .partition([1, score_tile_width_i32]);
    let generics = vec![
        score_tile_width_i32.to_string(),
        depth_tile_i32.to_string(),
        query_sequence_len_i32.to_string(),
        query_head_count_i32.to_string(),
        key_value_head_count_i32.to_string(),
        head_dim_i32.to_string(),
    ];
    let (_query, _key, score_output, _scale) = cutile_attention_kernel::attention_scores_async(
        query_tensor.device_operation(),
        key_tensor.device_operation(),
        score_output,
        scale.device_operation(),
    )
    .generics(generics)
    .await
    .map_err(|error| {
        AttentionError::DeviceError(format!("cutile attention score launch failed: {error:?}"))
    })?;
    let score_tensor = Arc::new(score_output.unpartition());

    let softmax_output = cutile::api::zeros::<2, f32>([rows, shape.key_value_sequence_len])
        .partition([1, key_sequence_len_i32]);
    let weight_tensor = if options.causal {
        let generics = vec![
            key_sequence_len_i32.to_string(),
            query_sequence_len_i32.to_string(),
            query_head_count_i32.to_string(),
        ];
        let (_scores, softmax_output, _offset) = cutile_attention_kernel::masked_row_softmax_async(
            score_tensor.device_operation(),
            softmax_output,
            query_position_offset_i32.device_operation(),
        )
        .generics(generics)
        .await
        .map_err(|error| {
            AttentionError::DeviceError(format!(
                "cutile masked attention softmax launch failed: {error:?}"
            ))
        })?;
        Arc::new(softmax_output.unpartition())
    } else {
        let (_scores, softmax_output) = cutile_attention_kernel::row_softmax_async(
            score_tensor.device_operation(),
            softmax_output,
        )
        .generics(vec![key_sequence_len_i32.to_string()])
        .await
        .map_err(|error| {
            AttentionError::DeviceError(format!(
                "cutile attention softmax launch failed: {error:?}"
            ))
        })?;
        Arc::new(softmax_output.unpartition())
    };

    let output =
        cutile::api::zeros::<2, f32>([rows, shape.head_dim]).partition([1, value_tile_i32]);
    let generics = vec![
        value_tile_i32.to_string(),
        value_tile_i32.to_string(),
        query_sequence_len_i32.to_string(),
        query_head_count_i32.to_string(),
        key_value_head_count_i32.to_string(),
        key_sequence_len_i32.to_string(),
    ];
    let (_weights, _value, output) = cutile_attention_kernel::attention_weighted_values_async(
        weight_tensor.device_operation(),
        value_tensor.device_operation(),
        output,
    )
    .generics(generics)
    .await
    .map_err(|error| {
        AttentionError::DeviceError(format!(
            "cutile attention weighted-value launch failed: {error:?}"
        ))
    })?;
    let output_shape = &[
        shape.batch_size,
        shape.query_sequence_len,
        shape.query_head_count,
        shape.head_dim,
    ];
    let output = output.unpartition().reshape_dyn(output_shape);
    Ok(Some(GpuTensor::from_cutile_tensor(output, output_shape)?))
}

async fn scaled_dot_product_attention_host_bridge(
    query: &GpuTensor,
    key: &GpuTensor,
    value: &GpuTensor,
    shape: AttentionShape,
    options: AttentionOptions,
) -> Result<GpuTensor, AttentionError> {
    let q = query.to_host_async().await?;
    let k = key.to_host_async().await?;
    let v = value.to_host_async().await?;
    let result = scaled_dot_product_attention_host(&q, &k, &v, shape, options)?;
    let output_shape = &[
        shape.batch_size,
        shape.query_sequence_len,
        shape.query_head_count,
        shape.head_dim,
    ];
    Ok(GpuTensor::from_host_async(&result, output_shape).await?)
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU scaled dot-product attention.
pub async fn scaled_dot_product_attention(
    query: &GpuTensor,
    key: &GpuTensor,
    value: &GpuTensor,
    shape: AttentionShape,
    options: AttentionOptions,
) -> Result<GpuTensor, AttentionError> {
    validate_tensor_lengths(query, key, value, shape)?;
    let scale = options.resolve_scale(shape.head_dim)?;
    #[cfg(not(target_os = "linux"))]
    let _ = scale;

    match backend_for_shape(shape, options) {
        #[cfg(target_os = "linux")]
        AttentionBackend::Cutile => {
            if let Some(output) =
                scaled_dot_product_attention_cutile(query, key, value, shape, options, scale)
                    .await?
            {
                return Ok(output);
            }
            scaled_dot_product_attention_host_bridge(query, key, value, shape, options).await
        }
        AttentionBackend::HostFallback => {
            scaled_dot_product_attention_host_bridge(query, key, value, shape, options).await
        }
        #[cfg(not(target_os = "linux"))]
        AttentionBackend::Cutile => {
            unreachable!("non-Linux platforms never select the cutile attention backend")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn approx_eq_slice(lhs: &[f32], rhs: &[f32]) {
        approx_eq_slice_with_tolerance(lhs, rhs, 1e-5);
    }

    /// Verifies that the attention kernel reports the active platform backend when queried. This catches accidental backend metadata regressions.
    #[test]
    fn reports_platform_attention_backend() {
        #[cfg(target_os = "linux")]
        let expected = [AttentionKernel {
            name: "grouped_query_attention",
            backend: AttentionBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [AttentionKernel {
            name: "grouped_query_attention",
            backend: AttentionBackend::HostFallback,
        }];

        assert_eq!(supported_attention_kernels(), expected);
    }

    /// Verifies that the attention backend selector only enables cutile for the supported Linux shape envelope. This catches accidental dispatch of unsupported shapes into the device kernels.
    #[test]
    fn cutile_support_heuristic_respects_shape_boundaries() {
        let supported = AttentionShape::new(1, 2, 16, 2, 1, 8);
        let bad_kv_len = AttentionShape::new(1, 2, 12, 2, 1, 8);
        let bad_head_dim = AttentionShape::new(1, 2, 16, 2, 1, 6);
        let options = AttentionOptions::default();

        #[cfg(target_os = "linux")]
        {
            assert_eq!(
                backend_for_shape(supported, options),
                AttentionBackend::Cutile
            );
            assert_eq!(
                backend_for_shape(bad_kv_len, options),
                AttentionBackend::HostFallback
            );
            assert_eq!(
                backend_for_shape(bad_head_dim, options),
                AttentionBackend::HostFallback
            );
        }

        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(
                backend_for_shape(supported, options),
                AttentionBackend::HostFallback
            );
            assert_eq!(
                backend_for_shape(bad_kv_len, options),
                AttentionBackend::HostFallback
            );
            assert_eq!(
                backend_for_shape(bad_head_dim, options),
                AttentionBackend::HostFallback
            );
        }
    }

    /// Verifies Q·Kᵀ dot-product scores with a single head and scale=1.
    ///
    /// This catches errors in the score accumulation loop or indexing.
    #[test]
    fn computes_attention_scores_for_single_head() {
        let shape = AttentionShape::new(1, 1, 2, 1, 1, 2);
        let scores = attention_scores_host(
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            shape,
            AttentionOptions {
                softmax_scale: Some(1.0),
                ..AttentionOptions::default()
            },
        )
        .unwrap();

        approx_eq_slice(&scores, &[1.0, 0.0]);
    }

    /// Verifies full scaled dot-product attention (scores → softmax_host → V weighted sum).
    ///
    /// This catches errors in softmax_host normalization or value accumulation.
    #[test]
    fn attends_over_single_head_without_masking() {
        let shape = AttentionShape::new(1, 1, 2, 1, 1, 2);
        let output = scaled_dot_product_attention_host(
            &[1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[10.0, 1.0, 1.0, 20.0],
            shape,
            AttentionOptions {
                softmax_scale: Some(1.0),
                ..AttentionOptions::default()
            },
        )
        .unwrap();

        approx_eq_slice(&output, &[7.5795274, 6.109886]);
    }

    /// Verifies that causal masking prevents attending to future positions.
    ///
    /// This catches off-by-one errors in the causal mask boundary.
    #[test]
    fn applies_causal_mask() {
        let shape = AttentionShape::new(1, 2, 2, 1, 1, 2);
        let output = scaled_dot_product_attention_host(
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[10.0, 0.0, 0.0, 5.0],
            shape,
            AttentionOptions {
                causal: true,
                softmax_scale: Some(1.0),
                ..AttentionOptions::default()
            },
        )
        .unwrap();

        approx_eq_slice(&output, &[10.0, 0.0, 2.6894143, 3.655293]);
    }

    /// Verifies that multiple query heads share a single KV head in GQA.
    ///
    /// This catches errors in the query_head / kv_group_size mapping.
    #[test]
    fn grouped_query_attention_reuses_key_value_heads() {
        let shape = AttentionShape::new(1, 1, 2, 2, 1, 1);
        let output = scaled_dot_product_attention_host(
            &[1.0, 2.0],
            &[1.0, 0.0],
            &[10.0, 1.0],
            shape,
            AttentionOptions {
                softmax_scale: Some(1.0),
                ..AttentionOptions::default()
            },
        )
        .unwrap();

        approx_eq_slice(&output, &[7.5795274, 8.927174]);
    }

    /// Verifies that query_position_offset shifts the causal window for decode-phase attention.
    ///
    /// This catches off-by-one errors in the offset + query_index causal boundary.
    #[test]
    fn query_position_offset_supports_decode_style_causal_attention() {
        let shape = AttentionShape::new(1, 1, 3, 1, 1, 1);
        let output = scaled_dot_product_attention_host(
            &[1.0],
            &[1.0, 1.0, 1.0],
            &[10.0, 20.0, 30.0],
            shape,
            AttentionOptions {
                causal: true,
                query_position_offset: 2,
                softmax_scale: Some(1.0),
            },
        )
        .unwrap();

        approx_eq_slice(&output, &[20.0]);
    }

    /// Verifies that non-divisible query/KV head counts are rejected.
    ///
    /// This catches missing validation of the GQA head grouping constraint.
    #[test]
    fn rejects_invalid_head_grouping() {
        let shape = AttentionShape::new(1, 1, 1, 3, 2, 1);
        let error = attention_scores_host(
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0],
            shape,
            AttentionOptions::default(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            AttentionError::HeadCountMismatch {
                query_head_count: 3,
                key_value_head_count: 2,
            }
        );
    }

    /// Verifies that a too-small output buffer is rejected.
    ///
    /// This catches missing output length validation in the _into variant.
    #[test]
    fn rejects_output_length_mismatch() {
        let shape = AttentionShape::new(1, 1, 1, 1, 1, 1);
        let mut output = [0.0; 0];
        let error = scaled_dot_product_attention_into_host(
            &[1.0],
            &[1.0],
            &[1.0],
            shape,
            AttentionOptions::default(),
            &mut output,
        )
        .unwrap_err();

        assert_eq!(
            error,
            AttentionError::LengthMismatch {
                argument: "output",
                expected: 1,
                actual: 0,
            }
        );
    }

    /// Verifies that a zero softmax_host scale is rejected as invalid.
    ///
    /// This catches missing scale validation (zero or negative scales produce wrong results).
    #[test]
    fn rejects_invalid_scale() {
        let shape = AttentionShape::new(1, 1, 1, 1, 1, 1);
        let error = attention_scores_host(
            &[1.0],
            &[1.0],
            shape,
            AttentionOptions {
                softmax_scale: Some(0.0),
                ..AttentionOptions::default()
            },
        )
        .unwrap_err();

        assert_eq!(error, AttentionError::InvalidScale(0.0));
    }

    /// Verifies that the async GPU attention matches the host fallback.
    /// This catches regressions in the GPU attention path.
    #[tokio::test]
    async fn gpu_attention_matches_host_fallback() {
        let shape = AttentionShape::new(1, 1, 3, 1, 1, 2);
        let options = AttentionOptions::default();
        let query = vec![1.0, 0.0];
        let key = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let value = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected =
            scaled_dot_product_attention_host(&query, &key, &value, shape, options).unwrap();
        let gpu_q = GpuTensor::from_host(&query, &[1, 1, 1, 2]).unwrap();
        let gpu_k = GpuTensor::from_host(&key, &[1, 3, 1, 2]).unwrap();
        let gpu_v = GpuTensor::from_host(&value, &[1, 3, 1, 2]).unwrap();
        let result = super::scaled_dot_product_attention(&gpu_q, &gpu_k, &gpu_v, shape, options)
            .await
            .unwrap();
        assert_eq!(result.shape(), &[1, 1, 1, 2]);
        assert_eq!(result.to_host(), expected);
    }

    /// Verifies that the async GPU attention matches host parity across supported non-causal and decode-causal shapes. This catches device-kernel math, masking, or reshape regressions while keeping cutile kernel compilation serialized in one test.
    #[tokio::test]
    async fn gpu_attention_matches_host_for_supported_cutile_shapes() {
        let shape = AttentionShape::new(1, 2, 16, 2, 1, 8);
        let options = AttentionOptions {
            softmax_scale: Some(0.5),
            ..AttentionOptions::default()
        };
        let query = (0..shape.query_len())
            .map(|index| (index % 11) as f32 * 0.1 - 0.5)
            .collect::<Vec<_>>();
        let key = (0..shape.key_len())
            .map(|index| (index % 7) as f32 * 0.125 - 0.25)
            .collect::<Vec<_>>();
        let value = (0..shape.value_len())
            .map(|index| (index % 13) as f32 * 0.2 - 0.6)
            .collect::<Vec<_>>();
        let expected =
            scaled_dot_product_attention_host(&query, &key, &value, shape, options).unwrap();
        let gpu_q = GpuTensor::from_host(
            &query,
            &[
                1,
                shape.query_sequence_len,
                shape.query_head_count,
                shape.head_dim,
            ],
        )
        .unwrap();
        let gpu_k = GpuTensor::from_host(
            &key,
            &[
                1,
                shape.key_value_sequence_len,
                shape.key_value_head_count,
                shape.head_dim,
            ],
        )
        .unwrap();
        let gpu_v = GpuTensor::from_host(
            &value,
            &[
                1,
                shape.key_value_sequence_len,
                shape.key_value_head_count,
                shape.head_dim,
            ],
        )
        .unwrap();

        #[cfg(target_os = "linux")]
        assert_eq!(backend_for_shape(shape, options), AttentionBackend::Cutile);

        let result = super::scaled_dot_product_attention(&gpu_q, &gpu_k, &gpu_v, shape, options)
            .await
            .unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 8]);
        approx_eq_slice_with_tolerance(&result.to_host(), &expected, 5e-5);

        let causal_shape = AttentionShape::new(1, 1, 16, 2, 1, 8);
        let causal_options = AttentionOptions {
            causal: true,
            query_position_offset: 7,
            softmax_scale: Some(0.75),
        };
        let causal_query = (0..causal_shape.query_len())
            .map(|index| (index % 5) as f32 * 0.3 - 0.6)
            .collect::<Vec<_>>();
        let causal_key = (0..causal_shape.key_len())
            .map(|index| (index % 9) as f32 * 0.15 - 0.3)
            .collect::<Vec<_>>();
        let causal_value = (0..causal_shape.value_len())
            .map(|index| (index % 17) as f32 * 0.05 - 0.4)
            .collect::<Vec<_>>();
        let causal_expected = scaled_dot_product_attention_host(
            &causal_query,
            &causal_key,
            &causal_value,
            causal_shape,
            causal_options,
        )
        .unwrap();
        let causal_gpu_q = GpuTensor::from_host(
            &causal_query,
            &[
                1,
                causal_shape.query_sequence_len,
                causal_shape.query_head_count,
                causal_shape.head_dim,
            ],
        )
        .unwrap();
        let causal_gpu_k = GpuTensor::from_host(
            &causal_key,
            &[
                1,
                causal_shape.key_value_sequence_len,
                causal_shape.key_value_head_count,
                causal_shape.head_dim,
            ],
        )
        .unwrap();
        let causal_gpu_v = GpuTensor::from_host(
            &causal_value,
            &[
                1,
                causal_shape.key_value_sequence_len,
                causal_shape.key_value_head_count,
                causal_shape.head_dim,
            ],
        )
        .unwrap();

        #[cfg(target_os = "linux")]
        assert_eq!(
            backend_for_shape(causal_shape, causal_options),
            AttentionBackend::Cutile
        );

        let causal_result = super::scaled_dot_product_attention(
            &causal_gpu_q,
            &causal_gpu_k,
            &causal_gpu_v,
            causal_shape,
            causal_options,
        )
        .await
        .unwrap();
        approx_eq_slice_with_tolerance(&causal_result.to_host(), &causal_expected, 5e-5);
    }
}
