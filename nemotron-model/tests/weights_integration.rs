use nemotron_model::{WeightManifest, WeightTensorDescriptor, SAFETENSORS_INDEX_FILE};
use serde_json::Map;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn test_root(name: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("nemotron-weights-{name}-{unique}"))
}

fn write_safetensors_file(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
    let mut offset = 0_usize;
    let mut header = Map::new();
    let mut data = Vec::new();

    for (name, dtype, shape, bytes) in tensors {
        let start = offset;
        offset += bytes.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": *dtype,
                "shape": *shape,
                "data_offsets": [start, offset],
            }),
        );
        data.extend_from_slice(bytes);
    }

    let header_bytes = serde_json::to_vec(&header).expect("header JSON should serialize");
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(&data);

    fs::write(path, file_bytes).expect("safetensors file should be written");
}

#[test]
fn weight_manifest_loads_indexed_fixture_and_tensor_bytes() {
    let root = test_root("indexed");
    fs::create_dir_all(&root).expect("test root should be created");
    write_safetensors_file(
        &root.join("model-00001-of-00002.safetensors"),
        &[("embed.weight", "F32", &[2], &[1, 2, 3, 4, 5, 6, 7, 8])],
    );
    write_safetensors_file(
        &root.join("model-00002-of-00002.safetensors"),
        &[("lm_head.weight", "U8", &[3], &[9, 10, 11])],
    );
    fs::write(
        root.join(SAFETENSORS_INDEX_FILE),
        serde_json::to_vec(&serde_json::json!({
            "weight_map": {
                "embed.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            }
        }))
        .expect("index JSON should serialize"),
    )
    .expect("index file should be written");

    let manifest = WeightManifest::from_root(&root).expect("manifest should load");

    assert_eq!(manifest.files.len(), 2);
    assert_eq!(
        manifest.tensor("lm_head.weight"),
        Some(&WeightTensorDescriptor {
            name: "lm_head.weight".to_string(),
            relative_path: PathBuf::from("model-00002-of-00002.safetensors"),
            dtype: "U8".to_string(),
            shape: vec![3],
            data_offsets: (0, 3),
        })
    );

    let tensor = manifest
        .load_tensor("embed.weight")
        .expect("tensor bytes should load");
    assert_eq!(tensor.shape, vec![2]);
    assert_eq!(tensor.data, vec![1, 2, 3, 4, 5, 6, 7, 8]);

    fs::remove_dir_all(root).expect("test root should be removed");
}
