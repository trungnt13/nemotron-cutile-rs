use crate::{
    AttentionLayer, AttentionLayerError, LayerCache, LayerStub, Mamba2Cache, Mamba2Error,
    Mamba2ForwardShape, Mamba2Mixer, MlpError, MlpLayer, MoeError, MoeLayer,
};
use nemotron_kernels::attention::AttentionOptions;
use nemotron_kernels::rms_norm::{rms_norm, RmsNormError};
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "block",
    summary: "Hybrid Nemotron block orchestration for residual layers.",
};

#[derive(Clone, Debug, PartialEq)]
pub enum BlockMixer {
    Attention(AttentionLayer),
    Mamba(Mamba2Mixer),
    Mlp(MlpLayer),
    Moe(MoeLayer),
}

#[derive(Clone, Debug, PartialEq)]
pub struct NemotronBlock {
    hidden_size: usize,
    norm_weight: Vec<f32>,
    epsilon: f32,
    mixer: BlockMixer,
}

impl NemotronBlock {
    pub fn new(
        hidden_size: usize,
        norm_weight: Vec<f32>,
        epsilon: f32,
        mixer: BlockMixer,
    ) -> Result<Self, BlockError> {
        if hidden_size == 0 {
            return Err(BlockError::InvalidHiddenSize(hidden_size));
        }

        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(BlockError::InvalidEpsilon(epsilon));
        }

        if norm_weight.len() != hidden_size {
            return Err(BlockError::LengthMismatch {
                argument: "norm_weight",
                expected: hidden_size,
                actual: norm_weight.len(),
            });
        }

        validate_mixer_hidden_size(&mixer, hidden_size)?;

        Ok(Self {
            hidden_size,
            norm_weight,
            epsilon,
            mixer,
        })
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn mixer(&self) -> &BlockMixer {
        &self.mixer
    }

    pub fn forward(
        &self,
        hidden_states: &[f32],
        row_count: usize,
        cache: Option<&mut LayerCache>,
    ) -> Result<Vec<f32>, BlockError> {
        let mut output = vec![0.0; row_count * self.hidden_size];
        self.forward_into(hidden_states, row_count, cache, &mut output)?;
        Ok(output)
    }

    pub fn forward_into(
        &self,
        hidden_states: &[f32],
        row_count: usize,
        cache: Option<&mut LayerCache>,
        output: &mut [f32],
    ) -> Result<(), BlockError> {
        if row_count == 0 {
            return Err(BlockError::InvalidRowCount(row_count));
        }

        let expected = row_count * self.hidden_size;
        if hidden_states.len() != expected {
            return Err(BlockError::LengthMismatch {
                argument: "hidden_states",
                expected,
                actual: hidden_states.len(),
            });
        }

        if output.len() != expected {
            return Err(BlockError::LengthMismatch {
                argument: "output",
                expected,
                actual: output.len(),
            });
        }

        let normalized = rms_norm_rows(hidden_states, &self.norm_weight, self.epsilon)
            .map_err(BlockError::Norm)?;
        let mixed = match (&self.mixer, cache) {
            (BlockMixer::Attention(layer), _) => layer
                .forward_self_attention(
                    &normalized,
                    1,
                    row_count,
                    AttentionOptions {
                        causal: true,
                        query_position_offset: 0,
                        softmax_scale: None,
                    },
                )
                .map_err(BlockError::Attention)?,
            (BlockMixer::Mlp(layer), _) => layer
                .forward(&normalized, row_count)
                .map_err(BlockError::Mlp)?,
            (BlockMixer::Moe(layer), _) => layer
                .forward(&normalized, row_count)
                .map_err(BlockError::Moe)?,
            (BlockMixer::Mamba(layer), Some(layer_cache)) => {
                let mamba_cache = ensure_mamba_cache(layer_cache, layer, row_count)?;
                layer
                    .forward(
                        &normalized,
                        Mamba2ForwardShape::new(1, row_count),
                        Some(mamba_cache),
                    )
                    .map_err(BlockError::Mamba)?
            }
            (BlockMixer::Mamba(layer), None) => layer
                .forward(&normalized, Mamba2ForwardShape::new(1, row_count), None)
                .map_err(BlockError::Mamba)?,
        };

        for ((slot, residual), mixed_value) in output
            .iter_mut()
            .zip(hidden_states.iter().copied())
            .zip(mixed.into_iter())
        {
            *slot = residual + mixed_value;
        }

        Ok(())
    }
}

fn validate_mixer_hidden_size(mixer: &BlockMixer, hidden_size: usize) -> Result<(), BlockError> {
    match mixer {
        BlockMixer::Attention(layer) if layer.hidden_size() != hidden_size => {
            Err(BlockError::MixerHiddenSizeMismatch {
                expected: hidden_size,
                actual: layer.hidden_size(),
            })
        }
        BlockMixer::Mamba(layer) if layer.hidden_size() != hidden_size => {
            Err(BlockError::MixerHiddenSizeMismatch {
                expected: hidden_size,
                actual: layer.hidden_size(),
            })
        }
        BlockMixer::Mlp(layer) if layer.shape().hidden_dim != hidden_size => {
            Err(BlockError::MixerHiddenSizeMismatch {
                expected: hidden_size,
                actual: layer.shape().hidden_dim,
            })
        }
        BlockMixer::Moe(layer) if layer.shape().hidden_dim != hidden_size => {
            Err(BlockError::MixerHiddenSizeMismatch {
                expected: hidden_size,
                actual: layer.shape().hidden_dim,
            })
        }
        _ => Ok(()),
    }
}

fn ensure_mamba_cache<'a>(
    layer_cache: &'a mut LayerCache,
    layer: &Mamba2Mixer,
    _row_count: usize,
) -> Result<&'a mut Mamba2Cache, BlockError> {
    match layer_cache {
        LayerCache::Empty => {
            *layer_cache = LayerCache::Mamba(Mamba2Cache::new_zeroed(
                1,
                layer.conv_channels(),
                layer.conv_kernel_size(),
                layer.state_size(),
            ));
            match layer_cache {
                LayerCache::Mamba(cache) => Ok(cache),
                _ => unreachable!(),
            }
        }
        LayerCache::Mamba(cache) => Ok(cache),
        LayerCache::Attention(_) => Err(BlockError::CacheTypeMismatch("attention")),
    }
}

fn rms_norm_rows(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>, RmsNormError> {
    let row_width = weight.len();
    let row_count = input.len() / row_width;
    let mut output = vec![0.0; input.len()];

    for row_index in 0..row_count {
        let start = row_index * row_width;
        let end = start + row_width;
        let normalized = rms_norm(&input[start..end], weight, epsilon)?;
        output[start..end].copy_from_slice(&normalized);
    }

    Ok(output)
}

#[derive(Debug, PartialEq)]
pub enum BlockError {
    InvalidHiddenSize(usize),
    InvalidEpsilon(f32),
    InvalidRowCount(usize),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    MixerHiddenSizeMismatch {
        expected: usize,
        actual: usize,
    },
    CacheTypeMismatch(&'static str),
    Norm(RmsNormError),
    Attention(AttentionLayerError),
    Mamba(Mamba2Error),
    Mlp(MlpError),
    Moe(MoeError),
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHiddenSize(hidden_size) => {
                write!(f, "hidden size must be non-zero, got {hidden_size}")
            }
            Self::InvalidEpsilon(epsilon) => {
                write!(f, "epsilon must be finite and non-negative, got {epsilon}")
            }
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
            Self::MixerHiddenSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "mixer hidden size mismatch: expected {expected}, got {actual}"
                )
            }
            Self::CacheTypeMismatch(cache_kind) => {
                write!(f, "cache type mismatch: found {cache_kind} cache")
            }
            Self::Norm(source) => write!(f, "norm failed: {source:?}"),
            Self::Attention(source) => write!(f, "attention failed: {source}"),
            Self::Mamba(source) => write!(f, "mamba failed: {source}"),
            Self::Mlp(source) => write!(f, "mlp failed: {source}"),
            Self::Moe(source) => write!(f, "moe failed: {source}"),
        }
    }
}

impl Error for BlockError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearProjection, MoeShape};

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

    fn identity_projection(size: usize) -> LinearProjection {
        let mut weights = vec![0.0; size * size];
        for index in 0..size {
            weights[index * size + index] = 1.0;
        }
        LinearProjection::new_dense_f32(size, size, weights, None).unwrap()
    }

    #[test]
    fn attention_block_adds_residual() {
        let attention = AttentionLayer::new(
            2,
            1,
            1,
            2,
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
            identity_projection(2),
        )
        .unwrap();
        let block =
            NemotronBlock::new(2, vec![1.0, 1.0], 1e-5, BlockMixer::Attention(attention)).unwrap();
        let hidden_states = [1.0, 0.0, 0.0, 1.0];

        let output = block.forward(&hidden_states, 2, None).unwrap();

        assert_eq!(output.len(), hidden_states.len());
        assert!(output
            .iter()
            .zip(hidden_states)
            .any(|(out, input)| (out - input).abs() > 1e-6));
    }

    #[test]
    fn mlp_block_adds_residual() {
        let mlp = MlpLayer::new_dense_relu2(1, 1, vec![1.0], None, vec![1.0], None).unwrap();
        let block = NemotronBlock::new(1, vec![1.0], 1e-5, BlockMixer::Mlp(mlp)).unwrap();

        let output = block.forward(&[2.0], 1, None).unwrap();
        approx_eq_slice(&output, &[3.0]);
    }

    #[test]
    fn moe_block_runs() {
        let router = LinearProjection::new_dense_f32(1, 1, vec![1.0], None).unwrap();
        let expert = MlpLayer::new_dense_relu2(1, 1, vec![1.0], None, vec![1.0], None).unwrap();
        let moe = MoeLayer::new(MoeShape::new(1, 1, 1), router, vec![expert], None).unwrap();
        let block = NemotronBlock::new(1, vec![1.0], 1e-5, BlockMixer::Moe(moe)).unwrap();

        let output = block.forward(&[2.0], 1, None).unwrap();
        approx_eq_slice(&output, &[2.7310567]);
    }

    #[test]
    fn rejects_hidden_size_mismatch() {
        let mlp = MlpLayer::new_dense_relu2(1, 1, vec![1.0], None, vec![1.0], None).unwrap();
        let error = NemotronBlock::new(2, vec![1.0, 1.0], 1e-5, BlockMixer::Mlp(mlp)).unwrap_err();

        assert_eq!(
            error,
            BlockError::MixerHiddenSizeMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }
}
