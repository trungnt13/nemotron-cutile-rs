use assert_cmd::Command;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate manifest should live under workspace root")
        .to_path_buf()
}

#[test]
fn validator_binary_passes_reference_fixtures() {
    let workspace_root = workspace_root();
    let assert = Command::cargo_bin("nemotron-validate")
        .expect("validator binary should build")
        .current_dir(&workspace_root)
        .arg("data/reference_kernels")
        .arg("data/reference_outputs")
        .assert()
        .success();

    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(stdout.contains("summary: 9/9 validations passed"), "{stdout}");
    assert!(
        stdout.contains("e2e fixtures validate deterministic synthetic runtime behavior only"),
        "{stdout}"
    );
    assert!(
        stdout.contains("e2e/constant_world_runtime/forward_and_generate_from_two_token_prompt: PASS"),
        "{stdout}"
    );
}
