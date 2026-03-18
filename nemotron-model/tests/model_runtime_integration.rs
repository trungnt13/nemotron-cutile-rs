use nemotron_model::{
    generation_preview, EmbeddingTable, GenerationRequest, ModelConfig, ModelRuntime,
    ModelTokenizer, NemotronModel, DEFAULT_TOKENIZER_FILE,
};
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

fn tiny_config() -> ModelConfig {
    let mut config = ModelConfig::default();
    config.hidden_size = 2;
    config.vocab_size = 3;
    config.num_hidden_layers = 0;
    config.hybrid_override_pattern.clear();
    config.eos_token_id = Some(2);
    config
}

fn constant_world_runtime() -> ModelRuntime {
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
        .expect("embedding table should build"),
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
        .expect("lm head should build"),
    }
}

#[test]
fn attach_runtime_enables_forward_pass_and_summary_updates() {
    let root = test_root("attach-runtime");
    fs::create_dir_all(&root).expect("test root should be created");
    write_test_tokenizer(&root.join(DEFAULT_TOKENIZER_FILE));

    let mut model = NemotronModel::with_tokenizer_from_model_root(tiny_config(), &root)
        .expect("model tokenizer should load");

    assert!(model.has_tokenizer());
    assert!(!model.has_runtime());
    assert!(model.summary().contains("tokenizer=loaded"));
    assert!(model.summary().contains("runtime=unloaded"));

    let preview = generation_preview(
        &model,
        &GenerationRequest {
            prompt: "hello".to_string(),
            max_new_tokens: 1,
            add_special_tokens: false,
            stop_on_eos: true,
        },
    );
    assert!(preview.contains("generation: runtime unavailable"));

    let runtime = constant_world_runtime();
    model.attach_runtime(runtime.clone());

    assert!(model.has_runtime());
    assert_eq!(model.runtime(), Some(&runtime));
    assert!(model.summary().contains("runtime=loaded"));

    let output = model
        .forward_tokens(&[1, 2])
        .expect("forward pass should succeed");
    approx_eq_slice(&output.hidden_states, &[1.4142065, 0.0, 0.0, 1.4142065]);
    approx_eq_slice(&output.logits, &[0.0, 0.0, 1.4142065, 0.0, 0.0, 1.4142065]);
    assert_eq!(
        model
            .predict_next_token(&[1, 2])
            .expect("prediction should succeed"),
        2
    );

    fs::remove_dir_all(root).expect("test root should be removed");
}

#[test]
fn generate_stops_on_eos_and_preview_reports_generated_text() {
    let root = test_root("generate");
    fs::create_dir_all(&root).expect("test root should be created");
    write_test_tokenizer(&root.join(DEFAULT_TOKENIZER_FILE));

    let tokenizer = ModelTokenizer::from_model_root(&root).expect("tokenizer should load");
    let model = NemotronModel::with_runtime_and_tokenizer(
        tiny_config(),
        constant_world_runtime(),
        tokenizer,
    );
    let request = GenerationRequest {
        prompt: "hello".to_string(),
        max_new_tokens: 5,
        add_special_tokens: false,
        stop_on_eos: true,
    };

    let result = model.generate(&request).expect("generation should succeed");

    assert_eq!(result.prompt_token_ids, vec![1]);
    assert_eq!(result.generated_token_ids, vec![2]);
    assert_eq!(result.all_token_ids, vec![1, 2]);
    assert_eq!(result.generated_text.as_deref(), Some("world"));

    let preview = generation_preview(&model, &request);
    assert!(preview.contains("status: tokenizer ready"));
    assert!(preview.contains("generated_token_ids: [2]"));
    assert!(preview.contains("all_token_ids: [1, 2]"));
    assert!(preview.contains("generated_text: Some(\"world\")"));

    fs::remove_dir_all(root).expect("test root should be removed");
}
