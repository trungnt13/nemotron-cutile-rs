use crate::model::{ModelForwardError, ModelTextError, NemotronModel};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub add_special_tokens: bool,
    pub stop_on_eos: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationResult {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub all_token_ids: Vec<u32>,
    pub generated_text: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum GenerationError {
    Text(ModelTextError),
    Forward(ModelForwardError),
}

impl GenerationRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            max_new_tokens: 32,
            add_special_tokens: true,
            stop_on_eos: true,
        }
    }
}

impl NemotronModel {
    pub fn generate(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationResult, GenerationError> {
        let prompt_token_ids = self
            .encode_with_special_tokens(&request.prompt, request.add_special_tokens)
            .map_err(GenerationError::Text)?;
        let mut all_token_ids = prompt_token_ids.clone();
        let mut generated_token_ids = Vec::new();
        let eos_token = self.special_token_ids().eos;

        for _ in 0..request.max_new_tokens {
            let next_token = self
                .predict_next_token(&all_token_ids)
                .map_err(GenerationError::Forward)?;
            all_token_ids.push(next_token);
            generated_token_ids.push(next_token);

            if request.stop_on_eos && Some(next_token) == eos_token {
                break;
            }
        }

        let generated_text = if self.has_tokenizer() {
            Some(
                self.decode_with_special_tokens(&generated_token_ids, true)
                    .map_err(GenerationError::Text)?,
            )
        } else {
            None
        };

        Ok(GenerationResult {
            prompt_token_ids,
            generated_token_ids,
            all_token_ids,
            generated_text,
        })
    }

    /// Async GPU generation loop. Uses `predict_next_token_gpu` for each step.
    pub async fn generate_gpu(
        &self,
        request: &GenerationRequest,
    ) -> Result<GenerationResult, GenerationError> {
        let prompt_token_ids = self
            .encode_with_special_tokens(&request.prompt, request.add_special_tokens)
            .map_err(GenerationError::Text)?;
        let mut all_token_ids = prompt_token_ids.clone();
        let mut generated_token_ids = Vec::new();
        let eos_token = self.special_token_ids().eos;

        for _ in 0..request.max_new_tokens {
            let next_token = self
                .predict_next_token_gpu(&all_token_ids)
                .await
                .map_err(GenerationError::Forward)?;
            all_token_ids.push(next_token);
            generated_token_ids.push(next_token);

            if request.stop_on_eos && Some(next_token) == eos_token {
                break;
            }
        }

        let generated_text = if self.has_tokenizer() {
            Some(
                self.decode_with_special_tokens(&generated_token_ids, true)
                    .map_err(GenerationError::Text)?,
            )
        } else {
            None
        };

        Ok(GenerationResult {
            prompt_token_ids,
            generated_token_ids,
            all_token_ids,
            generated_text,
        })
    }
}

pub fn generation_preview(model: &NemotronModel, request: &GenerationRequest) -> String {
    let tokenizer_status =
        match model.encode_with_special_tokens(&request.prompt, request.add_special_tokens) {
            Ok(token_ids) => format!(
                "prompt_token_count: {}\nprompt_token_ids: {:?}\nstatus: tokenizer ready",
                token_ids.len(),
                token_ids
            ),
            Err(ModelTextError::MissingTokenizer) => "status: tokenizer not loaded".to_string(),
            Err(error) => format!("status: tokenizer error ({error})"),
        };

    let generation_status = match model.generate(request) {
        Ok(result) => format!(
            "generated_token_ids: {:?}\nall_token_ids: {:?}\ngenerated_text: {:?}",
            result.generated_token_ids, result.all_token_ids, result.generated_text
        ),
        Err(GenerationError::Text(ModelTextError::MissingTokenizer)) => {
            "generation: tokenizer unavailable".to_string()
        }
        Err(GenerationError::Forward(ModelForwardError::MissingRuntime)) => {
            "generation: runtime unavailable".to_string()
        }
        Err(error) => format!("generation: error ({error:?})"),
    };

    format!(
        "{}\nprompt: {}\nmax_new_tokens: {}\n{}\n{}",
        model.summary(),
        request.prompt,
        request.max_new_tokens,
        tokenizer_status,
        generation_status
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::model::{EmbeddingTable, ModelRuntime};
    use crate::{ModelTokenizer, DEFAULT_TOKENIZER_FILE};
    use nemotron_nn::LinearProjection;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer;

    fn test_root(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("nemotron-model-{name}-{unique}"))
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

    fn tiny_config() -> ModelConfig {
        let mut config = ModelConfig::default();
        config.hidden_size = 2;
        config.vocab_size = 3;
        config.num_hidden_layers = 0;
        config.hybrid_override_pattern.clear();
        config.eos_token_id = Some(2);
        config
    }

    fn tiny_runtime() -> ModelRuntime {
        ModelRuntime {
            embeddings: EmbeddingTable::new(
                3,
                2,
                vec![
                    1.0, 1.0, //
                    1.0, 0.0, //
                    0.0, 1.0,
                ],
            )
            .unwrap(),
            blocks: Vec::new(),
            final_norm_weight: vec![1.0, 1.0],
            lm_head: LinearProjection::new_dense_f32(
                2,
                3,
                vec![
                    0.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0,
                ],
                None,
            )
            .unwrap(),
        }
    }

    fn tiny_runtime_model() -> NemotronModel {
        NemotronModel::with_runtime(tiny_config(), tiny_runtime())
    }

    /// Verifies that `generation_preview` gracefully reports both "tokenizer not loaded"
    /// and "tokenizer unavailable" when the model has neither tokenizer nor runtime.
    ///
    /// This catches regressions in the preview's error-reporting branches.
    #[test]
    fn generation_preview_reports_missing_runtime_without_tokenizer() {
        let preview = generation_preview(&NemotronModel::default(), &GenerationRequest::new("hi"));
        assert!(preview.contains("tokenizer not loaded"));
        assert!(preview.contains("tokenizer unavailable"));
    }

    /// Verifies that `generate` returns `MissingTokenizer` when the model has a runtime
    /// but no tokenizer, since prompt encoding requires the tokenizer.
    ///
    /// This catches regressions where generate might bypass the tokenizer check.
    #[test]
    fn generate_rejects_missing_tokenizer() {
        let model = tiny_runtime_model();
        let result = model
            .generate(&GenerationRequest {
                prompt: String::new(),
                max_new_tokens: 2,
                add_special_tokens: false,
                stop_on_eos: false,
            })
            .unwrap_err();

        assert_eq!(
            result,
            GenerationError::Text(ModelTextError::MissingTokenizer)
        );
    }

    /// Verifies that `generate_gpu` produces the same EOS-stopped result as `generate`
    /// when the runtime always predicts the EOS token.
    ///
    /// This catches regressions where the async GPU loop diverges from host token
    /// accumulation, stopping, or decode behavior.
    #[tokio::test]
    async fn generate_gpu_matches_host_generation() {
        let root = test_root("generate-gpu");
        fs::create_dir_all(&root).expect("test root should be created");
        write_test_tokenizer(&root.join(DEFAULT_TOKENIZER_FILE));

        let tokenizer = ModelTokenizer::from_model_root(&root).expect("tokenizer should load");
        let model =
            NemotronModel::with_runtime_and_tokenizer(tiny_config(), tiny_runtime(), tokenizer);
        let request = GenerationRequest {
            prompt: "hello".to_string(),
            max_new_tokens: 5,
            add_special_tokens: false,
            stop_on_eos: true,
        };

        let host_result = model.generate(&request).unwrap();
        let gpu_result = model.generate_gpu(&request).await.unwrap();

        assert_eq!(gpu_result, host_result);

        fs::remove_dir_all(root).expect("test root should be removed");
    }

    /// Verifies that `generate_gpu` preserves the host forward error for an empty prompt
    /// when special-token insertion is disabled.
    ///
    /// This catches regressions where the GPU path reports a device-shape error instead of
    /// the same model-forward error the host generation loop returns.
    #[tokio::test]
    async fn generate_gpu_preserves_empty_prompt_error() {
        let model = tiny_runtime_model();
        let request = GenerationRequest {
            prompt: String::new(),
            max_new_tokens: 2,
            add_special_tokens: false,
            stop_on_eos: false,
        };

        let host_error = model.generate(&request).unwrap_err();
        let gpu_error = model.generate_gpu(&request).await.unwrap_err();

        assert_eq!(gpu_error, host_error);
    }
}
