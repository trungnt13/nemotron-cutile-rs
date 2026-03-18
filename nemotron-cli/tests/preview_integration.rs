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

#[test]
fn preview_uses_default_prompt_without_args() {
    assert_preview(
        run_cli(std::iter::empty::<&str>()),
        "Hello from nemotron-rs",
    );
}

#[test]
fn preview_echoes_custom_prompt() {
    assert_preview(run_cli(["preview status prompt"]), "preview status prompt");
}

#[test]
fn preview_ignores_additional_arguments_after_first_prompt() {
    assert_preview(
        run_cli(["first prompt wins", "ignored trailing arg", "--flag-like"]),
        "first prompt wins",
    );
}
