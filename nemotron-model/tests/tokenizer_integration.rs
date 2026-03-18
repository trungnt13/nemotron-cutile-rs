use nemotron_model::{
    generation_preview, GenerationRequest, ModelConfig, ModelTextError, ModelTokenizer,
    NemotronModel, SpecialTokenIds, DEFAULT_TOKENIZER_FILE,
};
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

/// Verifies that `ModelTokenizer::from_model_root` loads a tokenizer.json and that
/// encode/decode round-trips correctly through the integration boundary.
///
/// Fixture: 3-vocab WordLevel tokenizer (UNK=0, hello=1, world=2).
/// This catches regressions in the model-root path resolution and tokenizer I/O.
#[test]
fn loads_model_tokenizer_from_model_root() {
    let root = test_root("load");
    fs::create_dir_all(&root).expect("test root should be created");
    let tokenizer_path = root.join(DEFAULT_TOKENIZER_FILE);
    write_test_tokenizer(&tokenizer_path);

    let tokenizer = ModelTokenizer::from_model_root(&root).expect("tokenizer should load");

    assert_eq!(tokenizer.spec().source, tokenizer_path);
    assert_eq!(
        tokenizer
            .encode_with_special_tokens("hello world", false)
            .expect("encoding should succeed"),
        vec![1, 2]
    );
    assert_eq!(
        tokenizer
            .decode_with_special_tokens(&[1, 2], true)
            .expect("decoding should succeed"),
        "hello world"
    );

    fs::remove_dir_all(root).expect("test root should be removed");
}

/// Verifies that `NemotronModel::with_tokenizer_from_model_root` correctly wires the
/// tokenizer for encode/decode and exposes special token ids from the config.
///
/// Fixture: 3-vocab WordLevel tokenizer with default ModelConfig (bos=1, eos=2, pad=0).
/// This catches regressions in the model-level tokenizer delegation and special token plumbing.
#[test]
fn nemotron_model_uses_attached_tokenizer_for_text_roundtrip() {
    let root = test_root("model");
    fs::create_dir_all(&root).expect("test root should be created");
    write_test_tokenizer(&root.join(DEFAULT_TOKENIZER_FILE));

    let model = NemotronModel::with_tokenizer_from_model_root(ModelConfig::default(), &root)
        .expect("model tokenizer should load");

    assert!(model.has_tokenizer());
    assert_eq!(
        model.special_token_ids(),
        SpecialTokenIds {
            bos: Some(1),
            eos: Some(2),
            pad: Some(0),
        }
    );
    assert_eq!(
        model
            .encode_with_special_tokens("hello world", false)
            .expect("encoding should succeed"),
        vec![1, 2]
    );
    assert_eq!(
        model
            .decode_with_special_tokens(&[1, 2], true)
            .expect("decoding should succeed"),
        "hello world"
    );

    fs::remove_dir_all(root).expect("test root should be removed");
}

/// Verifies that `generation_preview` reports "tokenizer not loaded" and that
/// `encode` returns `MissingTokenizer` when no tokenizer is attached.
///
/// This catches regressions in the preview's no-tokenizer branch and the encode guard.
#[test]
fn generation_preview_reports_tokenizer_state() {
    let model = NemotronModel::default();
    let preview = generation_preview(&model, &GenerationRequest::new("hello"));

    assert!(preview.contains("status: tokenizer not loaded"));
    assert_eq!(
        model
            .encode("hello")
            .expect_err("encoding without tokenizer should fail"),
        ModelTextError::MissingTokenizer
    );
}
