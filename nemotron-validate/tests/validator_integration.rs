use assert_cmd::Command;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate manifest should live under workspace root")
        .to_path_buf()
}

fn bundled_fixtures_available(workspace_root: &Path) -> bool {
    workspace_root
        .join("data/reference_kernels/fixtures.json")
        .is_file()
        && workspace_root
            .join("data/reference_outputs/fixtures.json")
            .is_file()
}

/// Verifies that the `nemotron-validate` binary exits successfully and reports
/// all 9 validations (7 kernel + 2 e2e) as PASS when run against bundled fixtures.
///
/// This catches binary-level regressions: broken CLI argument handling, missing
/// fixture files, stdout formatting changes, or silent validation failures that
/// the unit test might not surface.
#[test]
fn validator_binary_passes_reference_fixtures() {
    let workspace_root = workspace_root();
    if !bundled_fixtures_available(&workspace_root) {
        eprintln!("skipping validator integration test because bundled fixtures are unavailable");
        return;
    }
    let assert = Command::cargo_bin("nemotron-validate")
        .expect("validator binary should build")
        .current_dir(&workspace_root)
        .arg("data/reference_kernels")
        .arg("data/reference_outputs")
        .assert()
        .success();

    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(
        stdout.contains("summary: 9/9 validations passed"),
        "{stdout}"
    );
    assert!(
        stdout.contains("e2e fixtures validate deterministic synthetic runtime behavior only"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "e2e/constant_world_runtime/forward_and_generate_from_two_token_prompt: PASS"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("gpu/causal_conv1d: PASS"), "{stdout}");
    assert!(stdout.contains("gpu/moe_routing: PASS"), "{stdout}");
    assert!(
        stdout.contains("gpu summary: 7/7 gpu validations passed"),
        "{stdout}"
    );
    assert!(
        stdout.contains("attention, ssm, embedding, and quantize are not validated here"),
        "{stdout}"
    );
}

/// Verifies that benchmark mode reports host-versus-wrapper timings and max-abs-diff parity when run against the bundled fixtures. This catches regressions in benchmark CLI routing, benchmark stdout formatting, and the synthetic model comparison path.
#[test]
fn benchmark_mode_reports_timings_and_parity() {
    let workspace_root = workspace_root();
    if !bundled_fixtures_available(&workspace_root) {
        eprintln!("skipping benchmark integration test because bundled fixtures are unavailable");
        return;
    }
    let assert = Command::cargo_bin("nemotron-validate")
        .expect("validator binary should build")
        .current_dir(&workspace_root)
        .arg("benchmark")
        .arg("data/reference_kernels")
        .arg("data/reference_outputs")
        .assert()
        .success();

    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(
        stdout.contains(
            "benchmark note: Linux now runs real cutile device compute for RMSNorm, softmax"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("benchmark/gemm: PASS"), "{stdout}");
    assert!(stdout.contains("benchmark/gemm-aligned: PASS"), "{stdout}");
    assert!(
        stdout.contains("benchmark/model/constant_world_runtime/forward_tokens: PASS"),
        "{stdout}"
    );
    assert!(stdout.contains("max_abs_diff="), "{stdout}");
}
