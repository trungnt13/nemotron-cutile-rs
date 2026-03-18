use crate::{ValidationReport, ValidationResult, DEFAULT_TOLERANCE};
use nemotron_model::{
    generation_preview, EmbeddingTable, GenerationRequest, ModelConfig, ModelRuntime,
    ModelTokenizer, NemotronModel,
};
use nemotron_nn::LinearProjection;
use serde::Deserialize;
use std::fs;
use std::path::Path;

const E2E_TOLERANCE: f32 = 1e-5;
const SYNTHETIC_LIMITATION_NOTE: &str =
    "e2e fixtures validate deterministic synthetic runtime behavior only; full Nemotron checkpoint parity is not covered yet.";

#[derive(Debug, Deserialize)]
struct E2eFixtureSet {
    notes: Option<String>,
    fixtures: Vec<E2eFixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct E2eFixture {
    name: String,
    description: Option<String>,
    limitations: Option<String>,
    tokenizer_file: String,
    config: E2eConfig,
    runtime: E2eRuntime,
    cases: Vec<E2eCase>,
}

#[derive(Clone, Debug, Deserialize)]
struct E2eConfig {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    #[serde(default)]
    hybrid_override_pattern: String,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct E2eRuntime {
    embeddings: Vec<f32>,
    final_norm_weight: Vec<f32>,
    lm_head_weights: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct E2eCase {
    name: String,
    prompt: String,
    add_special_tokens: bool,
    max_new_tokens: usize,
    stop_on_eos: bool,
    expected_prompt_token_ids: Vec<u32>,
    expected_hidden_states: Vec<f32>,
    expected_logits: Vec<f32>,
    expected_next_token: u32,
    expected_generated_token_ids: Vec<u32>,
    expected_all_token_ids: Vec<u32>,
    expected_generated_text: Option<String>,
    #[serde(default)]
    expected_preview_substrings: Vec<String>,
}

impl E2eConfig {
    fn to_model_config(&self) -> ModelConfig {
        let mut config = ModelConfig::default();
        config.hidden_size = self.hidden_size;
        config.vocab_size = self.vocab_size;
        config.num_hidden_layers = self.num_hidden_layers;
        config.hybrid_override_pattern = self.hybrid_override_pattern.clone();
        config.bos_token_id = self.bos_token_id;
        config.eos_token_id = self.eos_token_id;
        config.pad_token_id = self.pad_token_id;
        config
    }
}

impl E2eRuntime {
    fn to_model_runtime(&self, config: &ModelConfig) -> Result<ModelRuntime, String> {
        let embeddings = EmbeddingTable::new(
            config.vocab_size,
            config.hidden_size,
            self.embeddings.clone(),
        )
        .map_err(|error| format!("invalid embedding table: {error}"))?;
        let lm_head = LinearProjection::new_dense_f32(
            config.hidden_size,
            config.vocab_size,
            self.lm_head_weights.clone(),
            None,
        )
        .map_err(|error| format!("invalid lm_head: {error}"))?;

        Ok(ModelRuntime {
            embeddings,
            blocks: Vec::new(),
            final_norm_weight: self.final_norm_weight.clone(),
            lm_head,
        })
    }
}

pub(crate) fn run_validation(reference_dir: &Path) -> Result<ValidationReport, String> {
    let fixtures_path = reference_dir.join("fixtures.json");
    let fixture_set: E2eFixtureSet = serde_json::from_slice(
        &fs::read(&fixtures_path)
            .map_err(|error| format!("failed to read {}: {error}", fixtures_path.display()))?,
    )
    .map_err(|error| format!("failed to parse {}: {error}", fixtures_path.display()))?;

    let mut notes = vec![SYNTHETIC_LIMITATION_NOTE.to_string()];
    if let Some(note) = fixture_set.notes {
        notes.push(note);
    }

    let mut results = Vec::new();
    for fixture in &fixture_set.fixtures {
        if let Some(limitations) = &fixture.limitations {
            notes.push(format!("{}: {limitations}", fixture.name));
        }
        results.extend(validate_fixture(reference_dir, fixture)?);
    }

    Ok(ValidationReport { results, notes })
}

fn validate_fixture(
    reference_dir: &Path,
    fixture: &E2eFixture,
) -> Result<Vec<ValidationResult>, String> {
    let tokenizer_path = reference_dir.join(&fixture.tokenizer_file);
    let tokenizer = ModelTokenizer::from_file(&tokenizer_path).map_err(|error| {
        format!(
            "failed to load {} for {}: {error}",
            tokenizer_path.display(),
            fixture.name
        )
    })?;
    let config = fixture.config.to_model_config();
    let runtime = fixture.runtime.to_model_runtime(&config)?;
    let model = NemotronModel::with_runtime_and_tokenizer(config, runtime, tokenizer);

    Ok(fixture
        .cases
        .iter()
        .map(|case| validate_case(&model, fixture, case))
        .collect())
}

fn validate_case(model: &NemotronModel, fixture: &E2eFixture, case: &E2eCase) -> ValidationResult {
    let name = format!("e2e/{}/{}", fixture.name, case.name);
    let detail_prefix = fixture
        .description
        .as_deref()
        .map(|description| format!("{description}; "))
        .unwrap_or_default();

    let outcome = (|| -> Result<String, String> {
        let prompt_token_ids = model
            .encode_with_special_tokens(&case.prompt, case.add_special_tokens)
            .map_err(|error| format!("encode failed: {error}"))?;
        if prompt_token_ids != case.expected_prompt_token_ids {
            return Err(format!(
                "prompt_token_ids mismatch: actual={prompt_token_ids:?} expected={:?}",
                case.expected_prompt_token_ids
            ));
        }

        let forward = model
            .forward_tokens(&prompt_token_ids)
            .map_err(|error| format!("forward failed: {error}"))?;
        assert_close(
            "hidden_states",
            &forward.hidden_states,
            &case.expected_hidden_states,
            E2E_TOLERANCE,
        )?;
        assert_close(
            "logits",
            &forward.logits,
            &case.expected_logits,
            E2E_TOLERANCE,
        )?;

        let next_token = model
            .predict_next_token(&prompt_token_ids)
            .map_err(|error| format!("predict_next_token failed: {error}"))?;
        if next_token != case.expected_next_token {
            return Err(format!(
                "next_token mismatch: actual={next_token} expected={}",
                case.expected_next_token
            ));
        }

        let request = GenerationRequest {
            prompt: case.prompt.clone(),
            max_new_tokens: case.max_new_tokens,
            add_special_tokens: case.add_special_tokens,
            stop_on_eos: case.stop_on_eos,
        };
        let generated = model
            .generate(&request)
            .map_err(|error| format!("generate failed: {error:?}"))?;
        if generated.prompt_token_ids != case.expected_prompt_token_ids {
            return Err(format!(
                "generated prompt_token_ids mismatch: actual={:?} expected={:?}",
                generated.prompt_token_ids, case.expected_prompt_token_ids
            ));
        }
        if generated.generated_token_ids != case.expected_generated_token_ids {
            return Err(format!(
                "generated_token_ids mismatch: actual={:?} expected={:?}",
                generated.generated_token_ids, case.expected_generated_token_ids
            ));
        }
        if generated.all_token_ids != case.expected_all_token_ids {
            return Err(format!(
                "all_token_ids mismatch: actual={:?} expected={:?}",
                generated.all_token_ids, case.expected_all_token_ids
            ));
        }
        if generated.generated_text != case.expected_generated_text {
            return Err(format!(
                "generated_text mismatch: actual={:?} expected={:?}",
                generated.generated_text, case.expected_generated_text
            ));
        }

        let preview = generation_preview(model, &request);
        for fragment in &case.expected_preview_substrings {
            if !preview.contains(fragment) {
                return Err(format!(
                    "preview missing fragment {fragment:?}: preview={preview:?}"
                ));
            }
        }

        Ok(format!(
            "{}prompt_tokens={} generated_tokens={} max_abs_diff<={}",
            detail_prefix,
            prompt_token_ids.len(),
            generated.generated_token_ids.len(),
            DEFAULT_TOLERANCE.min(E2E_TOLERANCE)
        ))
    })();

    match outcome {
        Ok(detail) => ValidationResult {
            name,
            passed: true,
            detail: Some(detail),
        },
        Err(detail) => ValidationResult {
            name,
            passed: false,
            detail: Some(format!("{detail_prefix}{detail}")),
        },
    }
}

fn assert_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }

    let mut max_diff = 0.0_f32;
    let mut max_index = 0_usize;
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual_value - expected_value).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }

    if max_diff > tolerance {
        Err(format!(
            "{label} max_abs_diff={max_diff:.6} at index {max_index} exceeds tolerance {tolerance:.6}"
        ))
    } else {
        Ok(())
    }
}
