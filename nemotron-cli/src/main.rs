use nemotron_model::{generation_preview, GenerationRequest, NemotronModel};

#[tokio::main]
async fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello from nemotron-rs".to_string());
    let model = NemotronModel::default();
    let request = GenerationRequest::new(prompt);

    println!("{}", generation_preview(&model, &request));
}
