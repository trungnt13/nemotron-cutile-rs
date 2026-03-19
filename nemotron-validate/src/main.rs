mod e2e;

use nemotron_kernels::activations::{relu2_host, silu_host};
use nemotron_kernels::conv1d::{depthwise_causal_conv1d_host, Conv1dShape};
use nemotron_kernels::gemm::{gemm_host, GemmShape};
use nemotron_kernels::moe_routing::{moe_route_softmax_host, MoeRoutingShape};
use nemotron_kernels::rms_norm::rms_norm_host;
use nemotron_kernels::softmax::softmax_host;
use nemotron_model::workspace_summary;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_KERNEL_REFERENCE_DIR: &str = "data/reference_kernels";
const DEFAULT_OUTPUT_REFERENCE_DIR: &str = "data/reference_outputs";
const DEFAULT_TOLERANCE: f32 = 1e-4;

fn main() {
    let mut args = std::env::args().skip(1);
    let kernel_reference_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_KERNEL_REFERENCE_DIR));
    let output_reference_dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_REFERENCE_DIR));

    println!("{}", workspace_summary());
    println!("kernel validation root: {}", kernel_reference_dir.display());
    println!("e2e validation root: {}", output_reference_dir.display());

    match run_validation(&kernel_reference_dir, &output_reference_dir) {
        Ok(report) => {
            for note in &report.notes {
                println!("note: {note}");
            }
            for result in &report.results {
                println!(
                    "{}: {}{}",
                    result.name,
                    if result.passed { "PASS" } else { "FAIL" },
                    result
                        .detail
                        .as_ref()
                        .map(|detail| format!(" ({detail})"))
                        .unwrap_or_default()
                );
            }

            let passed = report.results.iter().filter(|result| result.passed).count();
            println!(
                "summary: {passed}/{} validations passed",
                report.results.len()
            );
            if !report.success() {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("validation error: {error}");
            std::process::exit(2);
        }
    }
}

#[derive(Clone, Debug)]
struct ValidationReport {
    results: Vec<ValidationResult>,
    notes: Vec<String>,
}

impl ValidationReport {
    fn success(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

#[derive(Clone, Debug)]
struct ValidationResult {
    name: String,
    passed: bool,
    detail: Option<String>,
}

fn run_validation(
    kernel_reference_dir: &Path,
    output_reference_dir: &Path,
) -> Result<ValidationReport, String> {
    let mut report = run_kernel_validation(kernel_reference_dir)?;
    let e2e_report = e2e::run_validation(output_reference_dir)?;
    report.results.extend(e2e_report.results);
    report.notes.extend(e2e_report.notes);
    Ok(report)
}

fn run_kernel_validation(reference_dir: &Path) -> Result<ValidationReport, String> {
    let fixtures_path = reference_dir.join("fixtures.json");
    let fixtures: BTreeMap<String, Value> = serde_json::from_slice(
        &fs::read(&fixtures_path)
            .map_err(|error| format!("failed to read {}: {error}", fixtures_path.display()))?,
    )
    .map_err(|error| format!("failed to parse {}: {error}", fixtures_path.display()))?;

    let results = vec![
        validate_gemm(fixtures.get("gemm").ok_or("missing gemm fixture")?)?,
        validate_rms_norm(fixtures.get("rms_norm").ok_or("missing rms_norm fixture")?)?,
        validate_softmax(fixtures.get("softmax").ok_or("missing softmax fixture")?)?,
        validate_silu(fixtures.get("silu").ok_or("missing silu fixture")?)?,
        validate_relu2(fixtures.get("relu2").ok_or("missing relu2 fixture")?)?,
        validate_conv1d(
            fixtures
                .get("causal_conv1d")
                .ok_or("missing causal_conv1d fixture")?,
        )?,
        validate_moe_routing(
            fixtures
                .get("moe_routing")
                .ok_or("missing moe_routing fixture")?,
        )?,
    ];

    Ok(ValidationReport {
        results,
        notes: Vec::new(),
    })
}

fn validate_gemm(fixture: &Value) -> Result<ValidationResult, String> {
    let a = flatten_f32(value_at(fixture, "a")?);
    let b = flatten_f32(value_at(fixture, "b")?);
    let expected = flatten_f32(value_at(fixture, "out")?);
    let a_rows = array_len(value_at(fixture, "a")?);
    let a_cols = array_len(&value_at(fixture, "a")?[0]);
    let b_cols = array_len(&value_at(fixture, "b")?[0]);
    let actual = gemm_host(&a, &b, GemmShape::new(a_rows, a_cols, b_cols))
        .map_err(|error| format!("{error:?}"))?;
    Ok(compare_f32("gemm_host", &actual, &expected, DEFAULT_TOLERANCE))
}

fn validate_rms_norm(fixture: &Value) -> Result<ValidationResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);
    let weight = flatten_f32(value_at(fixture, "weight")?);
    let expected = flatten_f32(value_at(fixture, "out")?);
    let epsilon = fixture
        .get("eps")
        .and_then(Value::as_f64)
        .ok_or("missing rms_norm eps")? as f32;
    let row_width = weight.len();
    let mut actual = Vec::with_capacity(x.len());
    for row in x.chunks_exact(row_width) {
        actual.extend(rms_norm_host(row, &weight, epsilon).map_err(|error| format!("{error:?}"))?);
    }
    Ok(compare_f32("rms_norm_host", &actual, &expected, 5e-4))
}

fn validate_softmax(fixture: &Value) -> Result<ValidationResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);
    let expected = flatten_f32(value_at(fixture, "out")?);
    let row_width = last_dim_len(value_at(fixture, "x")?)?;
    let mut actual = Vec::with_capacity(x.len());
    for row in x.chunks_exact(row_width) {
        actual.extend(softmax_host(row));
    }
    Ok(compare_f32("softmax_host", &actual, &expected, 5e-4))
}

fn validate_silu(fixture: &Value) -> Result<ValidationResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);
    let expected = flatten_f32(value_at(fixture, "out")?);
    Ok(compare_f32("silu_host", &silu_host(&x), &expected, DEFAULT_TOLERANCE))
}

fn validate_relu2(fixture: &Value) -> Result<ValidationResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);
    let expected = flatten_f32(value_at(fixture, "out")?);
    Ok(compare_f32(
        "relu2_host",
        &relu2_host(&x),
        &expected,
        DEFAULT_TOLERANCE,
    ))
}

fn validate_conv1d(fixture: &Value) -> Result<ValidationResult, String> {
    let input = flatten_f32(value_at(fixture, "x_bsd")?);
    let weights = flatten_f32(value_at(fixture, "weight_d1k")?);
    let bias = flatten_f32(value_at(fixture, "bias_d")?);
    let expected = flatten_f32(value_at(fixture, "out_bsd")?);
    let batch = array_len(value_at(fixture, "x_bsd")?);
    let sequence_len = array_len(&value_at(fixture, "x_bsd")?[0]);
    let channels = last_dim_len(value_at(fixture, "x_bsd")?)?;
    let kernel_size = fixture
        .get("kernel_size")
        .and_then(Value::as_u64)
        .ok_or("missing kernel_size")? as usize;

    let mut actual = vec![0.0; input.len()];
    let batch_stride = sequence_len * channels;
    let flattened_weights = reshape_conv_weights(&weights, channels, kernel_size)?;
    for batch_index in 0..batch {
        let start = batch_index * batch_stride;
        let end = start + batch_stride;
        let mut batch_output = depthwise_causal_conv1d_host(
            &input[start..end],
            &flattened_weights,
            Conv1dShape::new(sequence_len, channels, kernel_size),
        )
        .map_err(|error| format!("{error:?}"))?;
        for row in batch_output.chunks_exact_mut(channels) {
            for (value, bias_value) in row.iter_mut().zip(bias.iter().copied()) {
                *value += bias_value;
            }
        }
        actual[start..end].copy_from_slice(&batch_output);
    }

    Ok(compare_f32("causal_conv1d", &actual, &expected, 5e-4))
}

fn validate_moe_routing(fixture: &Value) -> Result<ValidationResult, String> {
    let logits = flatten_f32(value_at(fixture, "router_logits")?);
    let expected_indices = flatten_usize(value_at(fixture, "topk_indices")?)?;
    let expected_weights = flatten_f32(value_at(fixture, "topk_weights")?);
    let token_count = array_len(value_at(fixture, "router_logits")?);
    let expert_count = last_dim_len(value_at(fixture, "router_logits")?)?;
    let top_k = fixture
        .get("top_k")
        .and_then(Value::as_u64)
        .ok_or("missing top_k")? as usize;
    let actual = moe_route_softmax_host(
        &logits,
        MoeRoutingShape::new(token_count, expert_count, top_k),
    )
    .map_err(|error| format!("{error:?}"))?;

    let indices_match = actual.indices == expected_indices;
    let weights_report = compare_f32("moe_routing", &actual.weights, &expected_weights, 5e-4);
    Ok(ValidationResult {
        name: "moe_routing".to_string(),
        passed: indices_match && weights_report.passed,
        detail: if indices_match && weights_report.passed {
            None
        } else {
            Some(format!(
                "indices_match={indices_match}, {}",
                weights_report
                    .detail
                    .unwrap_or_else(|| "weights mismatch".to_string())
            ))
        },
    })
}

fn compare_f32(
    name: impl Into<String>,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> ValidationResult {
    let name = name.into();
    if actual.len() != expected.len() {
        return ValidationResult {
            name,
            passed: false,
            detail: Some(format!(
                "length mismatch: actual={} expected={}",
                actual.len(),
                expected.len()
            )),
        };
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

    ValidationResult {
        name,
        passed: max_diff <= tolerance,
        detail: if max_diff <= tolerance {
            Some(format!("max_abs_diff={max_diff:.6}"))
        } else {
            Some(format!("max_abs_diff={max_diff:.6} at index {max_index}"))
        },
    }
}

fn value_at<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("missing key {key}"))
}

fn array_len(value: &Value) -> usize {
    value.as_array().map(|array| array.len()).unwrap_or(0)
}

fn last_dim_len(value: &Value) -> Result<usize, String> {
    let mut current = value;
    while let Some(array) = current.as_array() {
        if array.is_empty() {
            return Err("encountered empty array while inferring shape".to_string());
        }
        if array.first().and_then(Value::as_array).is_none() {
            return Ok(array.len());
        }
        current = &array[0];
    }
    Err("expected nested array".to_string())
}

fn flatten_f32(value: &Value) -> Vec<f32> {
    let mut values = Vec::new();
    flatten_f32_into(value, &mut values);
    values
}

fn flatten_f32_into(value: &Value, output: &mut Vec<f32>) {
    if let Some(array) = value.as_array() {
        for item in array {
            flatten_f32_into(item, output);
        }
    } else if let Some(number) = value.as_f64() {
        output.push(number as f32);
    }
}

fn flatten_usize(value: &Value) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    flatten_usize_into(value, &mut values)?;
    Ok(values)
}

fn flatten_usize_into(value: &Value, output: &mut Vec<usize>) -> Result<(), String> {
    if let Some(array) = value.as_array() {
        for item in array {
            flatten_usize_into(item, output)?;
        }
        Ok(())
    } else if let Some(number) = value.as_u64() {
        output.push(number as usize);
        Ok(())
    } else {
        Err("expected integer array".to_string())
    }
}

fn reshape_conv_weights(
    weights: &[f32],
    channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, String> {
    if weights.len() != channels * kernel_size {
        return Err(format!(
            "conv weights length mismatch: actual={} expected={}",
            weights.len(),
            channels * kernel_size
        ));
    }
    Ok(weights.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that all bundled kernel and e2e reference fixtures pass validation
    /// when loaded from the repository's `data/` directory.
    ///
    /// This catches regressions in kernel implementations (GEMM, RMSNorm, softmax,
    /// SiLU, ReLU², Conv1D, MoE routing) and end-to-end model behavior (tokenization,
    /// forward pass, generation) against known-good reference outputs.
    #[test]
    fn bundled_reference_fixtures_validate() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let report = run_validation(
            &repo_root.join(DEFAULT_KERNEL_REFERENCE_DIR),
            &repo_root.join(DEFAULT_OUTPUT_REFERENCE_DIR),
        )
        .expect("bundled validation should load");

        assert!(
            report.success(),
            "validation results should pass: {report:?}"
        );
        assert!(
            report
                .results
                .iter()
                .any(|result| result.name.starts_with("e2e/")),
            "expected at least one e2e validation result"
        );
    }
}
