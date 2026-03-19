use crate::{array_len, flatten_f32, last_dim_len, value_at, DEFAULT_TOLERANCE};
use nemotron_kernels::activations::{relu2, relu2_host, silu, silu_host};
use nemotron_kernels::gemm::{gemm, gemm_host, GemmShape};
use nemotron_kernels::rms_norm::{rms_norm, rms_norm_host};
use nemotron_kernels::softmax::{softmax, softmax_host};
use nemotron_kernels::tensor::GpuTensor;
use nemotron_model::{
    EmbeddingTable, ModelConfig, ModelForwardOutput, ModelRuntime, NemotronModel,
};
use nemotron_nn::LinearProjection;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::path::Path;
use std::time::Instant;

const BENCHMARK_ITERATIONS: usize = 25;
const E2E_TOLERANCE: f32 = 1e-5;
const GPU_WRAPPER_NOTE: &str = "GPU wrapper paths currently delegate to host kernels, so these timings measure transfer/wrapper overhead plus parity rather than real GPU compute speed.";
const MODEL_BENCHMARK_NOTE: &str = "Model-level benchmark uses the bundled synthetic e2e runtime to compare forward_tokens against forward_tokens_gpu.";

#[derive(Clone, Debug)]
pub(crate) struct BenchmarkReport {
    pub(crate) results: Vec<BenchmarkResult>,
    pub(crate) notes: Vec<String>,
}

impl BenchmarkReport {
    pub(crate) fn success(&self) -> bool {
        self.results.iter().all(|result| result.passed)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BenchmarkResult {
    pub(crate) name: String,
    pub(crate) passed: bool,
    pub(crate) host_avg_ms: f64,
    pub(crate) gpu_wrapper_avg_ms: f64,
    pub(crate) max_abs_diff: f32,
    pub(crate) iterations: usize,
}

pub(crate) async fn run_benchmark_comparison(
    kernel_reference_dir: &Path,
    output_reference_dir: &Path,
) -> Result<BenchmarkReport, String> {
    let kernel_fixtures = load_kernel_fixtures(kernel_reference_dir)?;
    let mut results = vec![
        benchmark_gemm(
            kernel_fixtures.get("gemm").ok_or("missing gemm fixture")?,
            BENCHMARK_ITERATIONS,
        )
        .await?,
        benchmark_rms_norm(
            kernel_fixtures
                .get("rms_norm")
                .ok_or("missing rms_norm fixture")?,
            BENCHMARK_ITERATIONS,
        )
        .await?,
        benchmark_softmax(
            kernel_fixtures
                .get("softmax")
                .ok_or("missing softmax fixture")?,
            BENCHMARK_ITERATIONS,
        )
        .await?,
        benchmark_silu(
            kernel_fixtures.get("silu").ok_or("missing silu fixture")?,
            BENCHMARK_ITERATIONS,
        )
        .await?,
        benchmark_relu2(
            kernel_fixtures
                .get("relu2")
                .ok_or("missing relu2 fixture")?,
            BENCHMARK_ITERATIONS,
        )
        .await?,
    ];
    results.push(benchmark_model_forward(output_reference_dir, BENCHMARK_ITERATIONS).await?);

    Ok(BenchmarkReport {
        results,
        notes: vec![
            GPU_WRAPPER_NOTE.to_string(),
            MODEL_BENCHMARK_NOTE.to_string(),
        ],
    })
}

fn load_kernel_fixtures(reference_dir: &Path) -> Result<BTreeMap<String, Value>, String> {
    let fixtures_path = reference_dir.join("fixtures.json");
    serde_json::from_slice(
        &fs::read(&fixtures_path)
            .map_err(|error| format!("failed to read {}: {error}", fixtures_path.display()))?,
    )
    .map_err(|error| format!("failed to parse {}: {error}", fixtures_path.display()))
}

async fn benchmark_gemm(fixture: &Value, iterations: usize) -> Result<BenchmarkResult, String> {
    let a = flatten_f32(value_at(fixture, "a")?);
    let b = flatten_f32(value_at(fixture, "b")?);
    let a_rows = array_len(value_at(fixture, "a")?);
    let a_cols = array_len(&value_at(fixture, "a")?[0]);
    let b_cols = array_len(&value_at(fixture, "b")?[0]);
    let shape = GemmShape::new(a_rows, a_cols, b_cols);

    let (host_output, host_avg_ms) = measure_sync(iterations, || {
        gemm_host(&a, &b, shape).map_err(|error| format!("{error:?}"))
    })?;
    let (gpu_output, gpu_wrapper_avg_ms) = measure_async(iterations, || {
        let a = &a;
        let b = &b;
        async move {
            let gpu_a =
                GpuTensor::from_host(a, &[a_rows, a_cols]).map_err(|error| format!("{error:?}"))?;
            let gpu_b =
                GpuTensor::from_host(b, &[a_cols, b_cols]).map_err(|error| format!("{error:?}"))?;
            let output = gemm(&gpu_a, &gpu_b, shape)
                .await
                .map_err(|error| format!("{error:?}"))?;
            Ok(output.to_host())
        }
    })
    .await?;

    build_result(
        "benchmark/gemm",
        host_output,
        gpu_output,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        DEFAULT_TOLERANCE,
        iterations,
    )
}

async fn benchmark_rms_norm(fixture: &Value, iterations: usize) -> Result<BenchmarkResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);
    let weight = flatten_f32(value_at(fixture, "weight")?);
    let epsilon = fixture
        .get("eps")
        .and_then(Value::as_f64)
        .ok_or("missing rms_norm eps")? as f32;
    let row_width = weight.len();

    let (host_output, host_avg_ms) = measure_sync(iterations, || {
        let mut actual = Vec::with_capacity(x.len());
        for row in x.chunks_exact(row_width) {
            actual.extend(
                rms_norm_host(row, &weight, epsilon).map_err(|error| format!("{error:?}"))?,
            );
        }
        Ok(actual)
    })?;
    let (gpu_output, gpu_wrapper_avg_ms) = measure_async(iterations, || {
        let x = &x;
        let weight = &weight;
        async move {
            let mut actual = Vec::with_capacity(x.len());
            for row in x.chunks_exact(row_width) {
                let gpu_row = GpuTensor::from_host(row, &[row_width])
                    .map_err(|error| format!("{error:?}"))?;
                let gpu_weight = GpuTensor::from_host(weight, &[row_width])
                    .map_err(|error| format!("{error:?}"))?;
                let output = rms_norm(&gpu_row, &gpu_weight, epsilon)
                    .await
                    .map_err(|error| format!("{error:?}"))?;
                actual.extend(output.to_host());
            }
            Ok(actual)
        }
    })
    .await?;

    build_result(
        "benchmark/rms_norm",
        host_output,
        gpu_output,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        5e-4,
        iterations,
    )
}

async fn benchmark_softmax(fixture: &Value, iterations: usize) -> Result<BenchmarkResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);
    let row_width = last_dim_len(value_at(fixture, "x")?)?;

    let (host_output, host_avg_ms) = measure_sync(iterations, || {
        let mut actual = Vec::with_capacity(x.len());
        for row in x.chunks_exact(row_width) {
            actual.extend(softmax_host(row));
        }
        Ok(actual)
    })?;
    let (gpu_output, gpu_wrapper_avg_ms) = measure_async(iterations, || {
        let x = &x;
        async move {
            let mut actual = Vec::with_capacity(x.len());
            for row in x.chunks_exact(row_width) {
                let gpu_row = GpuTensor::from_host(row, &[row_width])
                    .map_err(|error| format!("{error:?}"))?;
                let output = softmax(&gpu_row)
                    .await
                    .map_err(|error| format!("{error:?}"))?;
                actual.extend(output.to_host());
            }
            Ok(actual)
        }
    })
    .await?;

    build_result(
        "benchmark/softmax",
        host_output,
        gpu_output,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        5e-4,
        iterations,
    )
}

async fn benchmark_silu(fixture: &Value, iterations: usize) -> Result<BenchmarkResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);

    let (host_output, host_avg_ms) = measure_sync(iterations, || Ok(silu_host(&x)))?;
    let (gpu_output, gpu_wrapper_avg_ms) = measure_async(iterations, || {
        let x = &x;
        async move {
            let gpu_x =
                GpuTensor::from_host(x, &[x.len()]).map_err(|error| format!("{error:?}"))?;
            let output = silu(&gpu_x).await.map_err(|error| format!("{error:?}"))?;
            Ok(output.to_host())
        }
    })
    .await?;

    build_result(
        "benchmark/silu",
        host_output,
        gpu_output,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        DEFAULT_TOLERANCE,
        iterations,
    )
}

async fn benchmark_relu2(fixture: &Value, iterations: usize) -> Result<BenchmarkResult, String> {
    let x = flatten_f32(value_at(fixture, "x")?);

    let (host_output, host_avg_ms) = measure_sync(iterations, || Ok(relu2_host(&x)))?;
    let (gpu_output, gpu_wrapper_avg_ms) = measure_async(iterations, || {
        let x = &x;
        async move {
            let gpu_x =
                GpuTensor::from_host(x, &[x.len()]).map_err(|error| format!("{error:?}"))?;
            let output = relu2(&gpu_x).await.map_err(|error| format!("{error:?}"))?;
            Ok(output.to_host())
        }
    })
    .await?;

    build_result(
        "benchmark/relu2",
        host_output,
        gpu_output,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        DEFAULT_TOLERANCE,
        iterations,
    )
}

async fn benchmark_model_forward(
    output_reference_dir: &Path,
    iterations: usize,
) -> Result<BenchmarkResult, String> {
    let fixture_set = load_model_fixture_set(output_reference_dir)?;
    let fixture = fixture_set
        .fixtures
        .first()
        .ok_or("missing e2e benchmark fixture")?;
    let case = fixture.cases.first().ok_or("missing e2e benchmark case")?;
    let config = fixture.config.to_model_config();
    let runtime = fixture.runtime.to_model_runtime(&config)?;
    let model = NemotronModel::with_runtime(config, runtime);
    let token_ids = case.expected_prompt_token_ids.clone();

    let (host_output, host_avg_ms) = measure_sync(iterations, || {
        model
            .forward_tokens(&token_ids)
            .map_err(|error| error.to_string())
    })?;
    let (gpu_output, gpu_wrapper_avg_ms) = measure_async(iterations, || {
        let token_ids = &token_ids;
        let model = &model;
        async move {
            model
                .forward_tokens_gpu(token_ids)
                .await
                .map_err(|error| error.to_string())
        }
    })
    .await?;
    let max_abs_diff = forward_output_max_abs_diff(&host_output, &gpu_output)?;

    Ok(BenchmarkResult {
        name: format!("benchmark/model/{}/forward_tokens", fixture.name),
        passed: max_abs_diff <= E2E_TOLERANCE,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        max_abs_diff,
        iterations,
    })
}

fn build_result(
    name: impl Into<String>,
    host_output: Vec<f32>,
    gpu_output: Vec<f32>,
    host_avg_ms: f64,
    gpu_wrapper_avg_ms: f64,
    tolerance: f32,
    iterations: usize,
) -> Result<BenchmarkResult, String> {
    let max_abs_diff = max_abs_diff(&host_output, &gpu_output)?;
    Ok(BenchmarkResult {
        name: name.into(),
        passed: max_abs_diff <= tolerance,
        host_avg_ms,
        gpu_wrapper_avg_ms,
        max_abs_diff,
        iterations,
    })
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(actual
        .iter()
        .zip(expected.iter())
        .map(|(actual_value, expected_value)| (actual_value - expected_value).abs())
        .fold(0.0_f32, f32::max))
}

fn forward_output_max_abs_diff(
    host_output: &ModelForwardOutput,
    gpu_output: &ModelForwardOutput,
) -> Result<f32, String> {
    let hidden_diff = max_abs_diff(&host_output.hidden_states, &gpu_output.hidden_states)?;
    let logits_diff = max_abs_diff(&host_output.logits, &gpu_output.logits)?;
    Ok(hidden_diff.max(logits_diff))
}

fn measure_sync<T, F>(iterations: usize, mut operation: F) -> Result<(T, f64), String>
where
    F: FnMut() -> Result<T, String>,
{
    if iterations == 0 {
        return Err("benchmark requires at least one iteration".to_string());
    }
    let _ = operation()?;
    let start = Instant::now();
    let mut last_result = None;
    for _ in 0..iterations {
        last_result = Some(operation()?);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
    let last_result = last_result.ok_or("benchmark requires at least one iteration")?;
    Ok((last_result, elapsed))
}

async fn measure_async<T, F, Fut>(iterations: usize, mut operation: F) -> Result<(T, f64), String>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, String>>,
{
    if iterations == 0 {
        return Err("benchmark requires at least one iteration".to_string());
    }
    let _ = operation().await?;
    let start = Instant::now();
    let mut last_result = None;
    for _ in 0..iterations {
        last_result = Some(operation().await?);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
    let last_result = last_result.ok_or("benchmark requires at least one iteration")?;
    Ok((last_result, elapsed))
}

#[derive(Debug, Deserialize)]
struct BenchmarkFixtureSet {
    fixtures: Vec<BenchmarkFixture>,
}

#[derive(Clone, Debug, Deserialize)]
struct BenchmarkFixture {
    name: String,
    config: BenchmarkConfig,
    runtime: BenchmarkRuntime,
    cases: Vec<BenchmarkCase>,
}

#[derive(Clone, Debug, Deserialize)]
struct BenchmarkConfig {
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
struct BenchmarkRuntime {
    embeddings: Vec<f32>,
    final_norm_weight: Vec<f32>,
    lm_head_weights: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct BenchmarkCase {
    expected_prompt_token_ids: Vec<u32>,
}

impl BenchmarkConfig {
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

impl BenchmarkRuntime {
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

fn load_model_fixture_set(reference_dir: &Path) -> Result<BenchmarkFixtureSet, String> {
    let fixtures_path = reference_dir.join("fixtures.json");
    serde_json::from_slice(
        &fs::read(&fixtures_path)
            .map_err(|error| format!("failed to read {}: {error}", fixtures_path.display()))?,
    )
    .map_err(|error| format!("failed to parse {}: {error}", fixtures_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Verifies that benchmark comparisons load the bundled fixtures and preserve host-vs-wrapper parity when the current wrapper path delegates back to host kernels. This catches regressions in benchmark fixture loading, timing harness execution, and max-diff reporting.
    #[tokio::test]
    async fn bundled_benchmark_comparison_runs() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let report = run_benchmark_comparison(
            &repo_root.join("data/reference_kernels"),
            &repo_root.join("data/reference_outputs"),
        )
        .await
        .expect("benchmark report should load");

        assert!(
            report.success(),
            "benchmark comparisons should pass: {report:?}"
        );
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("delegate to host kernels")),
            "expected wrapper note in benchmark report"
        );
        assert!(
            report
                .results
                .iter()
                .any(|result| result.name == "benchmark/gemm"),
            "expected gemm benchmark result"
        );
        assert!(
            report
                .results
                .iter()
                .any(|result| result.name.starts_with("benchmark/model/")),
            "expected model benchmark result"
        );
    }
}
