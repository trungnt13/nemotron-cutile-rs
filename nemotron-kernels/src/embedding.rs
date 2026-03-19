use crate::KernelStub;
use crate::tensor::{GpuTensor, TensorError};

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
    validate_shape(table, shape)?;

    if output.len() != shape.output_len(token_ids.len()) {
        return Err(EmbeddingError::LengthMismatch {
            argument: "output",
            expected: shape.output_len(token_ids.len()),
            actual: output.len(),
        });
    }

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

fn validate_shape(table: &[f32], shape: EmbeddingShape) -> Result<(), EmbeddingError> {
    if shape.vocab_size == 0 || shape.hidden_size == 0 {
        return Err(EmbeddingError::InvalidShape(shape));
    }

    if table.len() != shape.table_len() {
        return Err(EmbeddingError::LengthMismatch {
            argument: "table",
            expected: shape.table_len(),
            actual: table.len(),
        });
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


// ---------------------------------------------------------------------------
// Async GPU API
// ---------------------------------------------------------------------------

/// Async GPU embedding lookup.
pub async fn embedding_lookup(
    table: &GpuTensor,
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<GpuTensor, EmbeddingError> {
    let table_data = table.to_host_async().await?;
    let result = embedding_lookup_host(&table_data, token_ids, shape)?;
    let output_shape = &[token_ids.len(), shape.hidden_size];
    Ok(GpuTensor::from_host_async(&result, output_shape).await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_table() -> [f32; 12] {
        [
            0.0, 0.1, 0.2, //
            1.0, 1.1, 1.2, //
            2.0, 2.1, 2.2, //
            3.0, 3.1, 3.2, //
        ]
    }

    /// Verifies that the embedding kernel reports HostFallback as its backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_embedding_kernels(),
            [EmbeddingKernel {
                name: "embedding_lookup_host",
                backend: EmbeddingBackend::HostFallback,
            }]
        );
    }

    /// Verifies that a single token ID retrieves the correct embedding row.
    ///
    /// This catches off-by-one errors in the row offset calculation.
    #[test]
    fn looks_up_single_token() {
        let output = embedding_lookup_token_host(&sample_table(), 2, EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![2.0, 2.1, 2.2]);
    }

    /// Verifies that multiple token IDs produce concatenated embedding rows.
    ///
    /// This catches indexing errors when writing multiple rows to the output.
    #[test]
    fn looks_up_multiple_tokens() {
        let output = embedding_lookup_host(&sample_table(), &[3, 1], EmbeddingShape::new(4, 3)).unwrap();
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
        let output = embedding_lookup_host(&sample_table(), &[], EmbeddingShape::new(4, 3)).unwrap();
        assert!(output.is_empty());
    }

    /// Verifies that zero vocab_size is rejected as an invalid shape.
    ///
    /// This catches missing dimension validation.
    #[test]
    fn rejects_invalid_shape() {
        let error = embedding_lookup_host(&sample_table(), &[0], EmbeddingShape::new(0, 3)).unwrap_err();
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
        let error =
            embedding_lookup_host(&sample_table()[..11], &[0], EmbeddingShape::new(4, 3)).unwrap_err();
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
        let error = embedding_lookup_host(&sample_table(), &[4], EmbeddingShape::new(4, 3)).unwrap_err();
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
        let error =
            embedding_lookup_token_into_host(&sample_table(), 1, EmbeddingShape::new(4, 3), &mut output)
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
        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = EmbeddingShape::new(3, 2);
        let token_ids = &[0, 2, 1];
        let expected = embedding_lookup_host(&table, token_ids, shape).unwrap();
        let gpu_table = GpuTensor::from_host(&table, &[3, 2]).unwrap();
        let result = super::embedding_lookup(&gpu_table, token_ids, shape)
            .await.unwrap();
        assert_eq!(result.to_host(), expected);
    }

}
