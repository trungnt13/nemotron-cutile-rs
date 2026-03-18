use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

pub const DEFAULT_TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerSpec {
    pub source: PathBuf,
}

impl TokenizerSpec {
    pub fn new(source: impl Into<PathBuf>) -> Self {
        Self {
            source: source.into(),
        }
    }

    pub fn from_model_root(root: impl AsRef<Path>) -> Self {
        Self::new(root.as_ref().join(DEFAULT_TOKENIZER_FILE))
    }
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self::new(DEFAULT_TOKENIZER_FILE)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum TokenizerError {
    Load { source: PathBuf, message: String },
    Encode { message: String },
    Decode { message: String },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { source, message } => {
                write!(
                    f,
                    "failed to load tokenizer from {}: {message}",
                    source.display()
                )
            }
            Self::Encode { message } => write!(f, "failed to encode text: {message}"),
            Self::Decode { message } => write!(f, "failed to decode token ids: {message}"),
        }
    }
}

impl Error for TokenizerError {}

#[derive(Clone)]
pub struct ModelTokenizer {
    spec: TokenizerSpec,
    inner: Tokenizer,
}

impl ModelTokenizer {
    pub fn load(spec: TokenizerSpec) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(&spec.source).map_err(|error| TokenizerError::Load {
            source: spec.source.clone(),
            message: error.to_string(),
        })?;

        Ok(Self { spec, inner })
    }

    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self, TokenizerError> {
        Self::load(TokenizerSpec::new(path))
    }

    pub fn from_model_root(root: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        Self::load(TokenizerSpec::from_model_root(root))
    }

    pub fn spec(&self) -> &TokenizerSpec {
        &self.spec
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.inner.id_to_token(token_id)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_with_special_tokens(text, true)
    }

    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError::Encode {
                message: error.to_string(),
            })
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        self.decode_with_special_tokens(token_ids, true)
    }

    pub fn decode_with_special_tokens(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| TokenizerError::Decode {
                message: error.to_string(),
            })
    }
}

impl fmt::Debug for ModelTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelTokenizer")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelTokenizer, TokenizerSpec};
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn test_tokenizer_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("nemotron-model-{name}-{unique}.json"))
    }

    fn write_test_tokenizer(path: &Path) {
        let vocab = HashMap::from([
            ("[UNK]".to_string(), 0u32),
            ("hello".to_string(), 1u32),
            ("world".to_string(), 2u32),
        ]);

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("word level tokenizer should build");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Whitespace::default());
        tokenizer
            .save(path, false)
            .expect("tokenizer should save to tokenizer.json");
    }

    #[test]
    fn loads_tokenizer_json_and_encodes_decodes_text() {
        let path = test_tokenizer_path("roundtrip");
        write_test_tokenizer(&path);

        let tokenizer = ModelTokenizer::from_file(&path).expect("tokenizer should load");

        assert_eq!(tokenizer.spec().source, path);
        assert_eq!(
            tokenizer
                .encode_with_special_tokens("hello world", false)
                .expect("encoding should succeed"),
            vec![1, 2]
        );
        assert_eq!(
            tokenizer.decode(&[1, 2]).expect("decoding should succeed"),
            "hello world"
        );
        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));
        assert_eq!(tokenizer.id_to_token(2).as_deref(), Some("world"));

        fs::remove_file(tokenizer.spec().source.clone()).expect("test tokenizer should be removed");
    }

    #[test]
    fn resolves_default_tokenizer_path_from_model_root() {
        let root = std::env::temp_dir().join("nemotron-model-root");
        let spec = TokenizerSpec::from_model_root(&root);

        assert_eq!(spec.source, root.join(DEFAULT_TOKENIZER_FILE));
    }
}
