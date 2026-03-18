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
    use nemotron_nn::LinearProjection;

    fn identity_projection(size: usize) -> LinearProjection {
        let mut weights = vec![0.0; size * size];
        for index in 0..size {
            weights[index * size + index] = 1.0;
        }
        LinearProjection::new_dense_f32(size, size, weights, None).unwrap()
    }

    fn tiny_runtime_model() -> NemotronModel {
        let mut config = ModelConfig::default();
        config.hidden_size = 2;
        config.vocab_size = 2;
        config.num_hidden_layers = 0;
        config.hybrid_override_pattern.clear();
        config.eos_token_id = Some(1);

        NemotronModel::with_runtime(
            config,
            ModelRuntime {
                embeddings: EmbeddingTable::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
                blocks: Vec::new(),
                final_norm_weight: vec![1.0, 1.0],
                lm_head: identity_projection(2),
            },
        )
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
}
