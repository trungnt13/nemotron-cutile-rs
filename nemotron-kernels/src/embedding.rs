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
    name: "embedding_lookup",
    backend: EmbeddingBackend::HostFallback,
};

pub fn supported_embedding_kernels() -> [EmbeddingKernel; 1] {
    [EMBEDDING_LOOKUP]
}

pub fn embedding_lookup(
    table: &[f32],
    token_ids: &[usize],
    shape: EmbeddingShape,
) -> Result<Vec<f32>, EmbeddingError> {
    let mut output = vec![0.0; shape.output_len(token_ids.len())];
    embedding_lookup_into(table, token_ids, shape, &mut output)?;
    Ok(output)
}

pub fn embedding_lookup_into(
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

pub fn embedding_lookup_token(
    table: &[f32],
    token_id: usize,
    shape: EmbeddingShape,
) -> Result<Vec<f32>, EmbeddingError> {
    let mut output = vec![0.0; shape.hidden_size];
    embedding_lookup_token_into(table, token_id, shape, &mut output)?;
    Ok(output)
}

pub fn embedding_lookup_token_into(
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_embedding_kernels(),
            [EmbeddingKernel {
                name: "embedding_lookup",
                backend: EmbeddingBackend::HostFallback,
            }]
        );
    }

    #[test]
    fn looks_up_single_token() {
        let output = embedding_lookup_token(&sample_table(), 2, EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![2.0, 2.1, 2.2]);
    }

    #[test]
    fn looks_up_multiple_tokens() {
        let output = embedding_lookup(&sample_table(), &[3, 1], EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![3.0, 3.1, 3.2, 1.0, 1.1, 1.2]);
    }

    #[test]
    fn repeated_tokens_repeat_rows() {
        let output =
            embedding_lookup(&sample_table(), &[2, 2, 0], EmbeddingShape::new(4, 3)).unwrap();
        assert_eq!(output, vec![2.0, 2.1, 2.2, 2.0, 2.1, 2.2, 0.0, 0.1, 0.2]);
    }

    #[test]
    fn lookup_into_writes_existing_buffer() {
        let mut output = [-1.0; 6];
        embedding_lookup_into(
            &sample_table(),
            &[1, 0],
            EmbeddingShape::new(4, 3),
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [1.0, 1.1, 1.2, 0.0, 0.1, 0.2]);
    }

    #[test]
    fn empty_token_ids_produce_empty_output() {
        let output = embedding_lookup(&sample_table(), &[], EmbeddingShape::new(4, 3)).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn rejects_invalid_shape() {
        let error = embedding_lookup(&sample_table(), &[0], EmbeddingShape::new(0, 3)).unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::InvalidShape(EmbeddingShape::new(0, 3))
        );
    }

    #[test]
    fn rejects_table_length_mismatch() {
        let error =
            embedding_lookup(&sample_table()[..11], &[0], EmbeddingShape::new(4, 3)).unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::LengthMismatch {
                argument: "table",
                expected: 12,
                actual: 11,
            }
        );
    }

    #[test]
    fn rejects_output_length_mismatch() {
        let mut output = [0.0; 5];
        let error = embedding_lookup_into(
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

    #[test]
    fn rejects_out_of_range_token() {
        let error = embedding_lookup(&sample_table(), &[4], EmbeddingShape::new(4, 3)).unwrap_err();
        assert_eq!(
            error,
            EmbeddingError::TokenOutOfRange {
                token_id: 4,
                vocab_size: 4,
            }
        );
    }

    #[test]
    fn rejects_single_token_output_length_mismatch() {
        let mut output = [0.0; 2];
        let error =
            embedding_lookup_token_into(&sample_table(), 1, EmbeddingShape::new(4, 3), &mut output)
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
}
