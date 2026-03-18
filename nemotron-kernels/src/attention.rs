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

pub const GROUPED_QUERY_ATTENTION: AttentionKernel = AttentionKernel {
    name: "grouped_query_attention",
    backend: AttentionBackend::HostFallback,
};

pub fn supported_attention_kernels() -> [AttentionKernel; 1] {
    [GROUPED_QUERY_ATTENTION]
}

pub fn attention_scores(
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

pub fn scaled_dot_product_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: AttentionShape,
    options: AttentionOptions,
) -> Result<Vec<f32>, AttentionError> {
    let mut output = vec![0.0; shape.output_len()];
    scaled_dot_product_attention_into(query, key, value, shape, options, &mut output)?;
    Ok(output)
}

pub fn scaled_dot_product_attention_into(
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

    let scores = attention_scores(query, key, shape, options)?;
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

#[derive(Clone, Copy, Debug, PartialEq)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Verifies that the attention kernel reports HostFallback as its backend.
    ///
    /// This catches accidental backend tag changes before GPU kernels exist.
    #[test]
    fn reports_host_fallback_backend_for_now() {
        assert_eq!(
            supported_attention_kernels(),
            [AttentionKernel {
                name: "grouped_query_attention",
                backend: AttentionBackend::HostFallback,
            }]
        );
    }

    /// Verifies Q·Kᵀ dot-product scores with a single head and scale=1.
    ///
    /// This catches errors in the score accumulation loop or indexing.
    #[test]
    fn computes_attention_scores_for_single_head() {
        let shape = AttentionShape::new(1, 1, 2, 1, 1, 2);
        let scores = attention_scores(
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

    /// Verifies full scaled dot-product attention (scores → softmax → V weighted sum).
    ///
    /// This catches errors in softmax normalization or value accumulation.
    #[test]
    fn attends_over_single_head_without_masking() {
        let shape = AttentionShape::new(1, 1, 2, 1, 1, 2);
        let output = scaled_dot_product_attention(
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
        let output = scaled_dot_product_attention(
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
        let output = scaled_dot_product_attention(
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
        let output = scaled_dot_product_attention(
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
        let error = attention_scores(
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
        let error = scaled_dot_product_attention_into(
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

    /// Verifies that a zero softmax scale is rejected as invalid.
    ///
    /// This catches missing scale validation (zero or negative scales produce wrong results).
    #[test]
    fn rejects_invalid_scale() {
        let shape = AttentionShape::new(1, 1, 1, 1, 1, 1);
        let error = attention_scores(
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
}
