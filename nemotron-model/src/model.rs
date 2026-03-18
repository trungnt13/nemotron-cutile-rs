use crate::config::{ModelConfig, SpecialTokenIds};
use crate::tokenizer::{ModelTokenizer, TokenizerError};
use nemotron_kernels::embedding::{embedding_lookup, EmbeddingError, EmbeddingShape};
use nemotron_kernels::rms_norm::{rms_norm, RmsNormError};
use nemotron_nn::{BlockError, HybridCache, LinearError, LinearProjection, NemotronBlock};
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingTable {
    shape: EmbeddingShape,
    values: Vec<f32>,
}

impl EmbeddingTable {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        values: Vec<f32>,
    ) -> Result<Self, ModelForwardError> {
        let shape = EmbeddingShape::new(vocab_size, hidden_size);
        if values.len() != shape.table_len() {
            return Err(ModelForwardError::LengthMismatch {
                argument: "embedding_table",
                expected: shape.table_len(),
                actual: values.len(),
            });
        }

        Ok(Self { shape, values })
    }

    pub const fn shape(&self) -> EmbeddingShape {
        self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelRuntime {
    pub embeddings: EmbeddingTable,
    pub blocks: Vec<NemotronBlock>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head: LinearProjection,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelForwardOutput {
    pub hidden_states: Vec<f32>,
    pub logits: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct NemotronModel {
    config: ModelConfig,
    tokenizer: Option<ModelTokenizer>,
    runtime: Option<ModelRuntime>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum ModelTextError {
    MissingTokenizer,
    Tokenizer(TokenizerError),
}

#[derive(Debug, PartialEq)]
pub enum ModelForwardError {
    MissingRuntime,
    InvalidTokenId(u32),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    Embedding(EmbeddingError),
    Block {
        layer_index: usize,
        source: BlockError,
    },
    FinalNorm(RmsNormError),
    LmHead(LinearError),
}

impl fmt::Display for ModelTextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTokenizer => f.write_str("model tokenizer is not loaded"),
            Self::Tokenizer(error) => error.fmt(f),
        }
    }
}

impl Error for ModelTextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MissingTokenizer => None,
            Self::Tokenizer(error) => Some(error),
        }
    }
}

impl From<TokenizerError> for ModelTextError {
    fn from(error: TokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

impl fmt::Display for ModelForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRuntime => f.write_str("model runtime is not loaded"),
            Self::InvalidTokenId(token_id) => {
                write!(f, "token id {token_id} is out of range for usize")
            }
            Self::LengthMismatch {
                argument,
                expected,
                actual,
            } => write!(
                f,
                "{argument} length mismatch: expected {expected}, got {actual}"
            ),
            Self::Embedding(source) => write!(f, "embedding lookup failed: {source:?}"),
            Self::Block {
                layer_index,
                source,
            } => {
                write!(f, "block {layer_index} failed: {source}")
            }
            Self::FinalNorm(source) => write!(f, "final norm failed: {source:?}"),
            Self::LmHead(source) => write!(f, "lm head failed: {source}"),
        }
    }
}

impl Error for ModelForwardError {}

impl NemotronModel {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            tokenizer: None,
            runtime: None,
        }
    }

    pub fn with_runtime(config: ModelConfig, runtime: ModelRuntime) -> Self {
        Self {
            config,
            tokenizer: None,
            runtime: Some(runtime),
        }
    }

    pub fn with_tokenizer(config: ModelConfig, tokenizer: ModelTokenizer) -> Self {
        Self {
            config,
            tokenizer: Some(tokenizer),
            runtime: None,
        }
    }

    pub fn with_runtime_and_tokenizer(
        config: ModelConfig,
        runtime: ModelRuntime,
        tokenizer: ModelTokenizer,
    ) -> Self {
        Self {
            config,
            tokenizer: Some(tokenizer),
            runtime: Some(runtime),
        }
    }

    pub fn with_tokenizer_file(
        config: ModelConfig,
        path: impl Into<PathBuf>,
    ) -> Result<Self, TokenizerError> {
        Ok(Self::with_tokenizer(
            config,
            ModelTokenizer::from_file(path)?,
        ))
    }

    pub fn with_tokenizer_from_model_root(
        config: ModelConfig,
        root: impl AsRef<Path>,
    ) -> Result<Self, TokenizerError> {
        Ok(Self::with_tokenizer(
            config,
            ModelTokenizer::from_model_root(root)?,
        ))
    }

    pub fn attach_runtime(&mut self, runtime: ModelRuntime) {
        self.runtime = Some(runtime);
    }

    pub fn runtime(&self) -> Option<&ModelRuntime> {
        self.runtime.as_ref()
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> Option<&ModelTokenizer> {
        self.tokenizer.as_ref()
    }

    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub fn has_runtime(&self) -> bool {
        self.runtime.is_some()
    }

    pub fn special_token_ids(&self) -> SpecialTokenIds {
        self.config.special_token_ids()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ModelTextError> {
        self.encode_with_special_tokens(text, true)
    }

    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, ModelTextError> {
        self.require_tokenizer()?
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(Into::into)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, ModelTextError> {
        self.decode_with_special_tokens(token_ids, true)
    }

    pub fn decode_with_special_tokens(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, ModelTextError> {
        self.require_tokenizer()?
            .decode_with_special_tokens(token_ids, skip_special_tokens)
            .map_err(Into::into)
    }

    pub fn forward_tokens(
        &self,
        token_ids: &[u32],
    ) -> Result<ModelForwardOutput, ModelForwardError> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or(ModelForwardError::MissingRuntime)?;
        let token_ids = token_ids
            .iter()
            .copied()
            .map(|token_id| {
                usize::try_from(token_id).map_err(|_| ModelForwardError::InvalidTokenId(token_id))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut hidden_states = embedding_lookup(
            runtime.embeddings.values(),
            &token_ids,
            runtime.embeddings.shape(),
        )
        .map_err(ModelForwardError::Embedding)?;

        let mut cache = HybridCache::new(runtime.blocks.len());
        for (layer_index, block) in runtime.blocks.iter().enumerate() {
            let layer_cache = cache
                .layer_mut(layer_index)
                .expect("layer index is in bounds");
            hidden_states = block
                .forward(&hidden_states, token_ids.len(), Some(layer_cache))
                .map_err(|source| ModelForwardError::Block {
                    layer_index,
                    source,
                })?;
        }

        let hidden_states = rms_norm_rows(
            &hidden_states,
            &runtime.final_norm_weight,
            self.config.hidden_size,
            1e-5,
        )
        .map_err(ModelForwardError::FinalNorm)?;
        let logits = runtime
            .lm_head
            .project(&hidden_states, token_ids.len())
            .map_err(ModelForwardError::LmHead)?;

        Ok(ModelForwardOutput {
            hidden_states,
            logits,
        })
    }

    pub fn predict_next_token(&self, token_ids: &[u32]) -> Result<u32, ModelForwardError> {
        let output = self.forward_tokens(token_ids)?;
        let vocab_size = self.config.vocab_size;
        let last_row = &output.logits[output.logits.len() - vocab_size..];
        let (index, _) = last_row
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("lm head always produces at least one logit");
        Ok(index as u32)
    }

    pub fn summary(&self) -> String {
        format!(
            "model={} layers={} hidden={} vocab={} tokenizer={} runtime={}",
            self.config.display_name(),
            self.config.num_hidden_layers,
            self.config.hidden_size,
            self.config.vocab_size,
            if self.has_tokenizer() {
                "loaded"
            } else {
                "unloaded"
            },
            if self.has_runtime() {
                "loaded"
            } else {
                "unloaded"
            }
        )
    }

    fn require_tokenizer(&self) -> Result<&ModelTokenizer, ModelTextError> {
        self.tokenizer
            .as_ref()
            .ok_or(ModelTextError::MissingTokenizer)
    }
}

fn rms_norm_rows(
    input: &[f32],
    weight: &[f32],
    row_width: usize,
    epsilon: f32,
) -> Result<Vec<f32>, RmsNormError> {
    let mut output = vec![0.0; input.len()];
    for row_index in 0..(input.len() / row_width) {
        let start = row_index * row_width;
        let end = start + row_width;
        let normalized = rms_norm(&input[start..end], weight, epsilon)?;
        output[start..end].copy_from_slice(&normalized);
    }
    Ok(output)
}

impl Default for NemotronModel {
    fn default() -> Self {
        Self::new(ModelConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nemotron_nn::{BlockMixer, NemotronBlock};

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
    fn forward_tokens_runs_tiny_runtime() {
        let mut config = ModelConfig::default();
        config.hidden_size = 2;
        config.vocab_size = 2;
        config.num_hidden_layers = 0;
        config.hybrid_override_pattern.clear();
        let runtime = ModelRuntime {
            embeddings: EmbeddingTable::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            blocks: Vec::new(),
            final_norm_weight: vec![1.0, 1.0],
            lm_head: identity_projection(2),
        };
        let model = NemotronModel::with_runtime(config, runtime);

        let output = model.forward_tokens(&[0, 1]).unwrap();

        approx_eq_slice(&output.hidden_states, &[1.4142065, 0.0, 0.0, 1.4142065]);
        approx_eq_slice(&output.logits, &[1.4142065, 0.0, 0.0, 1.4142065]);
        assert_eq!(model.predict_next_token(&[0, 1]).unwrap(), 1);
    }

    #[test]
    fn block_stack_is_applied() {
        let mut config = ModelConfig::default();
        config.hidden_size = 1;
        config.vocab_size = 1;
        config.num_hidden_layers = 1;
        config.hybrid_override_pattern = "E".to_string();
        let block = NemotronBlock::new(
            1,
            vec![1.0],
            1e-5,
            BlockMixer::Mlp(
                nemotron_nn::MlpLayer::new_dense_relu2(1, 1, vec![1.0], None, vec![1.0], None)
                    .unwrap(),
            ),
        )
        .unwrap();
        let runtime = ModelRuntime {
            embeddings: EmbeddingTable::new(1, 1, vec![2.0]).unwrap(),
            blocks: vec![block],
            final_norm_weight: vec![1.0],
            lm_head: LinearProjection::new_dense_f32(1, 1, vec![1.0], None).unwrap(),
        };
        let model = NemotronModel::with_runtime(config, runtime);

        let output = model.forward_tokens(&[0]).unwrap();
        approx_eq_slice(&output.logits, &[0.99999875]);
    }

    #[test]
    fn rejects_missing_runtime() {
        let model = NemotronModel::default();
        assert_eq!(
            model.forward_tokens(&[0]).unwrap_err(),
            ModelForwardError::MissingRuntime
        );
    }
}
