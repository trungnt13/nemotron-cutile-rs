use crate::tensor::{GpuTensor, TensorError};
use crate::KernelStub;

pub const SPEC: KernelStub = KernelStub {
    name: "embedding",
    summary: "Embedding lookup kernels for token ids.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingBackend {
    HostFallback,
    Cutile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingKernel {
    pub name: &'static str,
    pub backend: EmbeddingBackend,
}

#[cfg(any(target_os = "linux", test))]
const CUTILE_BLOCK_SIZES: [usize; 9] = [256, 128, 64, 32, 16, 8, 4, 2, 1];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingShape {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl EmbeddingShape {
    pub const fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            vocab_size,
            hidden_size,
        }
    }

    pub const fn table_len(self) -> usize {
        self.vocab_size * self.hidden_size
    }

    pub const fn output_len(self, token_count: usize) -> usize {
        token_count * self.hidden_size
    }
}

#[cfg(target_os = "linux")]
pub const EMBEDDING_LOOKUP: EmbeddingKernel = EmbeddingKernel {
    name: "embedding_lookup_cutile",
    backend: EmbeddingBackend::Cutile,
};

#[cfg(not(target_os = "linux"))]
pub const EMBEDDING_LOOKUP: EmbeddingKernel = EmbeddingKernel {
    name: "embedding_lookup_host",
    backend: EmbeddingBackend::HostFallback,
};

pub fn supported_embedding_kernels() -> [EmbeddingKernel; 1] {
    [EMBEDDING_LOOKUP]
}

pub fn embedding_lookup_host(
    table: &[f32],
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<Vec<f32>, EmbeddingError> {
    let mut output = vec![0.0; shape.output_len(token_ids.len())];
    embedding_lookup_into_host(table, token_ids, shape, &mut output)?;
    Ok(output)
}

pub fn embedding_lookup_into_host(
    table: &[f32],
    token_ids: &[usize],
    shape: EmbeddingShape,
    output: &mut [f32],
) -> Result<(), EmbeddingError> {
    validate_lookup(table, token_ids, shape, output)?;

    for (token_index, token_id) in token_ids.iter().copied().enumerate() {
        let src_start = token_id * shape.hidden_size;
        let src_end = src_start + shape.hidden_size;
        let dst_start = token_index * shape.hidden_size;
        let dst_end = dst_start + shape.hidden_size;

        output[dst_start..dst_end].copy_from_slice(&table[src_start..src_end]);
    }

    Ok(())
}

pub fn embedding_lookup_token_host(
    table: &[f32],
    token_id: usize,
    shape: EmbeddingShape,
) -> Result<Vec<f32>, EmbeddingError> {
    let mut output = vec![0.0; shape.hidden_size];
    embedding_lookup_token_into_host(table, token_id, shape, &mut output)?;
    Ok(output)
}

pub fn embedding_lookup_token_into_host(
    table: &[f32],
    token_id: usize,
    shape: EmbeddingShape,
    output: &mut [f32],
) -> Result<(), EmbeddingError> {
    validate_shape(table, shape)?;

    if output.len() != shape.hidden_size {
        return Err(EmbeddingError::LengthMismatch {
            argument: "output",
            expected: shape.hidden_size,
            actual: output.len(),
        });
    }

    if token_id >= shape.vocab_size {
        return Err(EmbeddingError::TokenOutOfRange {
            token_id,
            vocab_size: shape.vocab_size,
        });
    }

    let start = token_id * shape.hidden_size;
    let end = start + shape.hidden_size;
    output.copy_from_slice(&table[start..end]);
    Ok(())
}

fn validate_lookup(
    table: &[f32],
    token_ids: &[usize],
    shape: EmbeddingShape,
    output: &mut [f32],
) -> Result<(), EmbeddingError> {
    validate_table_len(table.len(), shape)?;

    if output.len() != shape.output_len(token_ids.len()) {
        return Err(EmbeddingError::LengthMismatch {
            argument: "output",
            expected: shape.output_len(token_ids.len()),
            actual: output.len(),
        });
    }

    validate_token_ids(token_ids, shape)
}

fn validate_shape(table: &[f32], shape: EmbeddingShape) -> Result<(), EmbeddingError> {
    validate_table_len(table.len(), shape)
}

fn validate_table_len(table_len: usize, shape: EmbeddingShape) -> Result<(), EmbeddingError> {
    if shape.vocab_size == 0 || shape.hidden_size == 0 {
        return Err(EmbeddingError::InvalidShape(shape));
    }

    if table_len != shape.table_len() {
        return Err(EmbeddingError::LengthMismatch {
            argument: "table",
            expected: shape.table_len(),
            actual: table_len,
        });
    }

    Ok(())
}

fn validate_tensor_lookup(
    table: &GpuTensor,
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<(), EmbeddingError> {
    validate_table_len(table.numel(), shape)?;
    validate_token_ids(token_ids, shape)
}

fn validate_token_ids(token_ids: &[usize], shape: EmbeddingShape) -> Result<(), EmbeddingError> {
    for token_id in token_ids.iter().copied() {
        if token_id >= shape.vocab_size {
            return Err(EmbeddingError::TokenOutOfRange {
                token_id,
                vocab_size: shape.vocab_size,
            });
        }
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmbeddingError {
    InvalidShape(EmbeddingShape),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    TokenOutOfRange {
        token_id: usize,
        vocab_size: usize,
    },
    DeviceError(String),
}

impl From<TensorError> for EmbeddingError {
    fn from(e: TensorError) -> Self {
        EmbeddingError::DeviceError(e.to_string())
    }
}

#[cfg(any(target_os = "linux", test))]
fn select_cutile_block_size(hidden_size: usize) -> usize {
    CUTILE_BLOCK_SIZES
        .into_iter()
        .find(|block_size| hidden_size % block_size == 0)
        .unwrap_or(1)
}

#[cfg(any(target_os = "linux", test))]
fn supports_cutile_embedding(token_ids: &[usize], shape: EmbeddingShape) -> bool {
    if token_ids.is_empty() || shape.hidden_size == 0 {
        return false;
    }

    let Some(table_len) = shape.vocab_size.checked_mul(shape.hidden_size) else {
        return false;
    };
    let Some(_output_len) = token_ids.len().checked_mul(shape.hidden_size) else {
        return false;
    };

    table_len <= i32::MAX as usize
        && shape.hidden_size <= i32::MAX as usize
        && token_ids.len() <= i32::MAX as usize
        && token_ids
            .iter()
            .all(|token_id| *token_id <= i32::MAX as usize)
}

#[cfg(target_os = "linux")]
fn active_backend(token_ids: &[usize], shape: EmbeddingShape) -> EmbeddingBackend {
    if supports_cutile_embedding(token_ids, shape) {
        EmbeddingBackend::Cutile
    } else {
        EmbeddingBackend::HostFallback
    }
}

#[cfg(not(target_os = "linux"))]
fn active_backend(_token_ids: &[usize], _shape: EmbeddingShape) -> EmbeddingBackend {
    EmbeddingBackend::HostFallback
}

#[cfg(target_os = "linux")]
#[cutile::module]
mod cutile_embedding_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn embedding_lookup<const BLOCK_SIZE: i32>(
        output_ptr: *mut f32,
        table_ptr: *mut f32,
        token_ids: &Tensor<i64, { [-1] }>,
        hidden_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block_shape: Shape<{ [BLOCK_SIZE] }> = Shape::<{ [BLOCK_SIZE] }> {
            dims: &[BLOCK_SIZE],
        };
        let tiles_per_row: i32 = hidden_size / BLOCK_SIZE;
        let row: i32 = pid.0 / tiles_per_row;
        let hidden_block: i32 = pid.0 % tiles_per_row;
        let token_part: Partition<i64, { [1] }> = token_ids.partition(const_shape![1]);
        let token_tile: Tile<i64, { [] }> = token_part.load([row]).reshape(const_shape![]);
        let token_id: Tile<i32, { [] }> = trunci(token_tile);
        let hidden_offset: Tile<i32, { [] }> = scalar_to_tile(hidden_block * BLOCK_SIZE);
        let hidden_size_tile: Tile<i32, { [] }> = scalar_to_tile(hidden_size);
        let row_offset: Tile<i32, { [] }> = token_id * hidden_size_tile + hidden_offset;
        let table_ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(table_ptr);
        let table_ptr_tile: PointerTile<*mut i64, { [] }> = ptr_to_ptr(table_ptr_tile);
        let table_addr: Tile<i64, { [] }> = ptr_to_int(table_ptr_tile);
        let table_addr: i64 = tile_to_scalar(table_addr);
        let table_addr: Tile<i64, { [BLOCK_SIZE] }> = broadcast_scalar(table_addr, block_shape);
        let row_offset: i32 = tile_to_scalar(row_offset);
        let row_offset: Tile<i32, { [BLOCK_SIZE] }> = broadcast_scalar(row_offset, block_shape);
        let lane_offsets: Tile<i32, { [BLOCK_SIZE] }> = iota(block_shape);
        let element_offsets: Tile<i32, { [BLOCK_SIZE] }> = row_offset + lane_offsets;
        let element_size: Tile<i64, { [BLOCK_SIZE] }> = constant(4i64, block_shape);
        let byte_offsets: Tile<i64, { [BLOCK_SIZE] }> = exti(element_offsets) * element_size;
        let row_ptrs: PointerTile<*mut f32, { [BLOCK_SIZE] }> =
            int_to_ptr(table_addr + byte_offsets);
        let (row_tile_1d, _token): (Tile<f32, { [BLOCK_SIZE] }>, Token) =
            load_ptr_tko(row_ptrs, "relaxed", "device", None, None, None, None);

        let output_offset: i32 = row * hidden_size + hidden_block * BLOCK_SIZE;
        let output_offset: Tile<i32, { [BLOCK_SIZE] }> =
            broadcast_scalar(output_offset, block_shape);
        let output_ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_ptr);
        let output_ptr_tile: PointerTile<*mut i64, { [] }> = ptr_to_ptr(output_ptr_tile);
        let output_addr: Tile<i64, { [] }> = ptr_to_int(output_ptr_tile);
        let output_addr: i64 = tile_to_scalar(output_addr);
        let output_addr: Tile<i64, { [BLOCK_SIZE] }> = broadcast_scalar(output_addr, block_shape);
        let output_byte_offsets: Tile<i64, { [BLOCK_SIZE] }> =
            exti(output_offset + lane_offsets) * element_size;
        let output_ptrs: PointerTile<*mut f32, { [BLOCK_SIZE] }> =
            int_to_ptr(output_addr + output_byte_offsets);
        let _store_token = store_ptr_tko(
            output_ptrs,
            row_tile_1d,
            "relaxed",
            "device",
            None,
            None,
            None,
        );
    }
}

// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

async fn embedding_lookup_host_bridge(
    table: &GpuTensor,
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<GpuTensor, EmbeddingError> {
    let table_data = table.to_host_async().await?;
    let result = embedding_lookup_host(&table_data, token_ids, shape)?;
    let output_shape = &[token_ids.len(), shape.hidden_size];
    Ok(GpuTensor::from_host_async(&result, output_shape).await?)
}

#[cfg(target_os = "linux")]
async fn embedding_lookup_cutile(
    table: &GpuTensor,
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<GpuTensor, EmbeddingError> {
    use std::sync::Arc;

    use cutile::api;
    use cutile::tile_kernel::{value, IntoDeviceOperation, TileKernel};

    let block_size = select_cutile_block_size(shape.hidden_size);
    let hidden_size_i32 = i32::try_from(shape.hidden_size).map_err(|_| {
        EmbeddingError::DeviceError("cutile embedding hidden size overflowed i32".to_string())
    })?;
    let token_ids_device = Arc::new(
        token_ids
            .iter()
            .copied()
            .map(|token_id| token_id as i64)
            .collect::<Vec<_>>(),
    );
    let token_tensor = api::copy_host_vec_to_device(&token_ids_device)
        .await
        .map_err(|error| {
            EmbeddingError::DeviceError(format!("cutile embedding token upload failed: {error:?}"))
        })?
        .reshape([token_ids.len()]);
    let token_tensor = Arc::new(token_tensor);
    let output_len = token_ids.len() * shape.hidden_size;
    let output_data = Arc::new(vec![0.0f32; output_len]);
    let output = api::copy_host_vec_to_device(&output_data)
        .await
        .map_err(|error| {
            EmbeddingError::DeviceError(format!(
                "cutile embedding output allocation failed: {error:?}"
            ))
        })?;
    let output_ptr = output.device_pointer();
    let table_ptr = table
        .cutile_tensor_for_shape(&[shape.vocab_size, shape.hidden_size])
        .await?
        .device_pointer();
    let generics = vec![block_size.to_string()];
    let tiles_per_row = shape.hidden_size / block_size;
    let grid_x = u32::try_from(token_ids.len() * tiles_per_row).map_err(|_| {
        EmbeddingError::DeviceError("cutile embedding grid overflowed u32".to_string())
    })?;
    let (_output_ptr, _table_ptr, _token_tensor, _hidden_size) = unsafe {
        cutile_embedding_kernel::embedding_lookup_async(
            value(output_ptr),
            value(table_ptr),
            token_tensor.device_operation(),
            value(hidden_size_i32),
        )
    }
    .grid((grid_x, 1, 1))
    .generics(generics)
    .await
    .map_err(|error| {
        EmbeddingError::DeviceError(format!("cutile embedding launch failed: {error:?}"))
    })?;
    let output = output.reshape([token_ids.len(), shape.hidden_size]);
    GpuTensor::from_cutile_tensor(output, &[token_ids.len(), shape.hidden_size])
        .map_err(EmbeddingError::from)
}

/// Async GPU embedding lookup.
pub async fn embedding_lookup(
    table: &GpuTensor,
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<GpuTensor, EmbeddingError> {
    validate_tensor_lookup(table, token_ids, shape)?;

    match active_backend(token_ids, shape) {
        #[cfg(target_os = "linux")]
        EmbeddingBackend::Cutile => embedding_lookup_cutile(table, token_ids, shape).await,
        EmbeddingBackend::HostFallback => {
            embedding_lookup_host_bridge(table, token_ids, shape).await
        }
        #[cfg(not(target_os = "linux"))]
        EmbeddingBackend::Cutile => unreachable!("cutile backend is unavailable on this platform"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    static GPU_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn sample_table() -> [f32; 12] {
        [
            0.0, 0.1, 0.2, //
            1.0, 1.1, 1.2, //
            2.0, 2.1, 2.2, //
            3.0, 3.1, 3.2, //
        ]
    }

    /// Verifies that the embedding kernel registry reports the active primary backend for the current platform.
    ///
    /// This catches accidental backend tag drift as Linux enables cutile while other platforms keep the host fallback.
    #[test]
    fn reports_platform_primary_backend() {
        #[cfg(target_os = "linux")]
        let expected = [EmbeddingKernel {
            name: "embedding_lookup_cutile",
            backend: EmbeddingBackend::Cutile,
        }];

        #[cfg(not(target_os = "linux"))]
        let expected = [EmbeddingKernel {
            name: "embedding_lookup_host",
            backend: EmbeddingBackend::HostFallback,
        }];

        assert_eq!(supported_embedding_kernels(), expected);
    }

    /// Verifies that the cutile block-size selector chooses divisors of the hidden size for row tiling.
    ///
    /// This catches launches that would otherwise require partial tiles or out-of-bounds row loads.
    #[test]
    fn selects_divisible_cutile_block_sizes() {
        assert_eq!(select_cutile_block_size(256), 256);
        assert_eq!(select_cutile_block_size(192), 64);
        assert_eq!(select_cutile_block_size(30), 2);
        assert_eq!(select_cutile_block_size(17), 1);
    }

    /// Verifies that the cutile dispatch heuristic only accepts shapes that fit the current launcher contract.
    ///
    /// This catches unsupported Linux cases accidentally routing into the raw-pointer kernel.
    #[test]
    fn cutile_support_heuristic_respects_shape_boundaries() {
        assert!(supports_cutile_embedding(
            &[0, 3],
            EmbeddingShape::new(4, 6)
        ));
        assert!(!supports_cutile_embedding(&[], EmbeddingShape::new(4, 6)));
        assert!(!supports_cutile_embedding(&[0], EmbeddingShape::new(4, 0)));
    }

    /// Verifies that a single token ID retrieves the correct embedding row.
    ///
    /// This catches off-by-one errors in the row offset calculation.
    #[test]
    fn looks_up_single_token() {
        let output =
            embedding_lookup_token_host(&sample_table(), 2, EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![2.0, 2.1, 2.2]);
    }

    /// Verifies that multiple token IDs produce concatenated embedding rows.
    ///
    /// This catches indexing errors when writing multiple rows to the output.
    #[test]
    fn looks_up_multiple_tokens() {
        let output =
            embedding_lookup_host(&sample_table(), &[3, 1], EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![3.0, 3.1, 3.2, 1.0, 1.1, 1.2]);
    }

    /// Verifies that duplicate token IDs produce duplicate embedding rows.
    ///
    /// This catches bugs where repeated lookups might alias or skip writes.
    #[test]
    fn repeated_tokens_repeat_rows() {
        let output =
            embedding_lookup_host(&sample_table(), &[2, 2, 0], EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![2.0, 2.1, 2.2, 2.0, 2.1, 2.2, 0.0, 0.1, 0.2]);
    }

    /// Verifies that the _into variant writes into a pre-allocated buffer.
    ///
    /// This catches bugs where _into silently re-allocates instead of writing in place.
    #[test]
    fn lookup_into_writes_existing_buffer() {
        let mut output = [-1.0; 6];
        embedding_lookup_into_host(
            &sample_table(),
            &[1, 0],
            EmbeddingShape::new(4, 3),
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [1.0, 1.1, 1.2, 0.0, 0.1, 0.2]);
    }

    /// Verifies that an empty token ID list produces an empty output without error.
    ///
    /// This catches panics on zero-length inputs.
    #[test]
    fn empty_token_ids_produce_empty_output() {
        let output =
            embedding_lookup_host(&sample_table(), &[], EmbeddingShape::new(4, 3)).unwrap();
        assert!(output.is_empty());
    }

    /// Verifies that zero vocab_size is rejected as an invalid shape.
    ///
    /// This catches missing dimension validation.
    #[test]
    fn rejects_invalid_shape() {
        let error =
            embedding_lookup_host(&sample_table(), &[0], EmbeddingShape::new(0, 3)).unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::InvalidShape(EmbeddingShape::new(0, 3))
        );
    }

    /// Verifies that a table shorter than vocab_size × hidden_size is rejected.
    ///
    /// This catches missing table length validation.
    #[test]
    fn rejects_table_length_mismatch() {
        let error = embedding_lookup_host(&sample_table()[..11], &[0], EmbeddingShape::new(4, 3))
            .unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::LengthMismatch {
                argument: "table",
                expected: 12,
                actual: 11,
            }
        );
    }

    /// Verifies that a too-small output buffer is rejected in the _into variant.
    ///
    /// This catches missing output length validation.
    #[test]
    fn rejects_output_length_mismatch() {
        let mut output = [0.0; 5];
        let error = embedding_lookup_into_host(
            &sample_table(),
            &[0, 1],
            EmbeddingShape::new(4, 3),
            &mut output,
        )
        .unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::LengthMismatch {
                argument: "output",
                expected: 6,
                actual: 5,
            }
        );
    }

    /// Verifies that a token ID equal to vocab_size is rejected as out of range.
    ///
    /// This catches off-by-one in the token ID bounds check.
    #[test]
    fn rejects_out_of_range_token() {
        let error =
            embedding_lookup_host(&sample_table(), &[4], EmbeddingShape::new(4, 3)).unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::TokenOutOfRange {
                token_id: 4,
                vocab_size: 4,
            }
        );
    }

    /// Verifies that the single-token _into variant rejects a wrong-sized output buffer.
    ///
    /// This catches missing output validation in the single-token code path.
    #[test]
    fn rejects_single_token_output_length_mismatch() {
        let mut output = [0.0; 2];
        let error = embedding_lookup_token_into_host(
            &sample_table(),
            1,
            EmbeddingShape::new(4, 3),
            &mut output,
        )
        .unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::LengthMismatch {
                argument: "output",
                expected: 3,
                actual: 2,
            }
        );
    }

    /// Verifies that the async GPU embedding lookup matches the host fallback.
    /// This catches regressions in the GPU embedding path.
    #[tokio::test]
    async fn gpu_embedding_matches_host_fallback() {
        #[cfg(target_os = "linux")]
        let _guard = GPU_TEST_MUTEX
            .lock()
            .unwrap_or_else(|error| error.into_inner());

        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = EmbeddingShape::new(3, 2);
        let token_ids = &[0, 2, 1];
        let expected = embedding_lookup_host(&table, token_ids, shape).unwrap();
        let gpu_table = GpuTensor::from_host(&table, &[3, 2]).unwrap();
        let result = super::embedding_lookup(&gpu_table, token_ids, shape)
            .await
            .unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_host(), expected);
    }

    /// Verifies that the async GPU embedding lookup matches host output when each row spans multiple cutile blocks.
    ///
    /// This catches row-offset mistakes when the device kernel has to stitch together several row tiles.
    #[tokio::test]
    async fn gpu_embedding_matches_host_for_multi_block_rows() {
        #[cfg(target_os = "linux")]
        let _guard = GPU_TEST_MUTEX
            .lock()
            .unwrap_or_else(|error| error.into_inner());

        let table = vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, //
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, //
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, //
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, //
        ];
        let shape = EmbeddingShape::new(4, 6);
        let token_ids = &[3, 1, 0];
        let expected = embedding_lookup_host(&table, token_ids, shape).unwrap();
        let gpu_table = GpuTensor::from_host(&table, &[4, 6]).unwrap();
        let result = super::embedding_lookup(&gpu_table, token_ids, shape)
            .await
            .unwrap();
        assert_eq!(result.shape(), &[3, 6]);
        assert_eq!(result.to_host(), expected);
    }

    /// Verifies that async GPU lookup preserves the current empty-token device error instead of inventing a new contract.
    ///
    /// This catches backend-dispatch changes that would silently alter the existing zero-length GPU behavior.
    #[tokio::test]
    async fn gpu_embedding_empty_tokens_preserve_current_error() {
        let gpu_table = GpuTensor::from_host(&sample_table(), &[4, 3]).unwrap();
        let error = super::embedding_lookup(&gpu_table, &[], EmbeddingShape::new(4, 3))
            .await
            .unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::DeviceError(
                "tensor shape dimension 0 must be greater than zero".to_string()
            )
        );
    }
}
