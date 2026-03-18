//! Integration tests for the `nemotron-cli` binary.
//!
//! These tests invoke the compiled binary as a subprocess and verify its
//! stdout/stderr output against the expected `generation_preview` format.
//! They cover default-prompt fallback, custom-prompt passthrough, and
//! argument-count handling.

use assert_cmd::cargo::cargo_bin;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{Command, Output};

fn cli_bin() -> PathBuf {
    cargo_bin("nemotron-cli")
}

fn run_cli<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(cli_bin())
        .args(args)
        .output()
        .expect("nemotron-cli should run")
}

fn assert_preview(output: Output, prompt: &str) {
    assert!(
        output.status.success(),
        "cli exited unsuccessfully: {output:?}"
    );
    assert!(
        output.stderr.is_empty(),
        "expected no stderr, got: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout should be valid utf-8"),
        format!(
            "model=NemotronHForCausalLM layers=52 hidden=2688 vocab=131072 tokenizer=unloaded runtime=unloaded\nprompt: {prompt}\nmax_new_tokens: 32\nstatus: tokenizer not loaded\ngeneration: tokenizer unavailable\n"
        )
    );
}

/// Verifies that the CLI falls back to the built-in default prompt when
/// invoked with no arguments.
///
/// This catches regressions in the `unwrap_or_else` default-prompt path.
#[test]
fn preview_uses_default_prompt_without_args() {
    assert_preview(
        run_cli(std::iter::empty::<&str>()),
        "Hello from nemotron-rs",
    );
}

/// Verifies that a user-supplied prompt is passed through to the generation
/// preview output verbatim.
///
/// This catches bugs where argument parsing drops or mutates the prompt.
#[test]
fn preview_echoes_custom_prompt() {
    assert_preview(run_cli(["preview status prompt"]), "preview status prompt");
}

/// Verifies that only the first CLI argument is used as the prompt when
/// multiple arguments are supplied.
///
/// This catches regressions where extra arguments are concatenated or cause
/// errors, ensuring `args().nth(1)` semantics are preserved.
#[test]
fn preview_ignores_additional_arguments_after_first_prompt() {
    assert_preview(
        run_cli(["first prompt wins", "ignored trailing arg", "--flag-like"]),
        "first prompt wins",
    );
}
