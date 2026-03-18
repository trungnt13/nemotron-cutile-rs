use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const DEFAULT_WEIGHT_ROOT: &str = "data/model_weights";
pub const SAFETENSORS_FILE_EXTENSION: &str = "safetensors";
pub const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightManifest {
    pub root: PathBuf,
    pub format: String,
    pub files: Vec<WeightFile>,
    pub tensors: BTreeMap<String, WeightTensorDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightFile {
    pub relative_path: PathBuf,
    pub tensor_names: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightTensorDescriptor {
    pub name: String,
    pub relative_path: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Default for WeightManifest {
    fn default() -> Self {
        Self {
            root: PathBuf::from(DEFAULT_WEIGHT_ROOT),
            format: SAFETENSORS_FILE_EXTENSION.to_string(),
            files: Vec::new(),
            tensors: BTreeMap::new(),
        }
    }
}

impl WeightManifest {
    pub fn from_root(root: impl Into<PathBuf>) -> Result<Self, WeightError> {
        let root = root.into();
        if !root.exists() {
            return Err(WeightError::RootNotFound(root));
        }

        let index_path = root.join(SAFETENSORS_INDEX_FILE);
        let indexed_files = if index_path.exists() {
            Some(read_weight_index(&index_path)?)
        } else {
            None
        };

        let discovered_files = if let Some(index) = indexed_files.as_ref() {
            index
                .weight_map
                .values()
                .map(PathBuf::from)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            discover_safetensors_files(&root)?
        };

        if discovered_files.is_empty() {
            return Err(WeightError::NoSafetensorsFiles(root));
        }

        let mut files = Vec::new();
        let mut tensors = BTreeMap::new();

        for relative_path in discovered_files {
            let absolute_path = root.join(&relative_path);
            let tensor_headers = read_safetensors_header(&absolute_path)?;
            let mut tensor_names = Vec::new();

            for (name, header) in tensor_headers {
                if let Some(index) = indexed_files.as_ref() {
                    if let Some(expected_path) = index.weight_map.get(&name) {
                        if Path::new(expected_path) != relative_path {
                            return Err(WeightError::IndexMismatch {
                                tensor: name,
                                expected: PathBuf::from(expected_path),
                                actual: relative_path.clone(),
                            });
                        }
                    }
                }

                tensor_names.push(name.clone());
                let previous = tensors.insert(
                    name.clone(),
                    WeightTensorDescriptor {
                        name,
                        relative_path: relative_path.clone(),
                        dtype: header.dtype,
                        shape: header.shape,
                        data_offsets: header.data_offsets,
                    },
                );
                if previous.is_some() {
                    return Err(WeightError::DuplicateTensorName(
                        tensor_names.last().cloned().unwrap(),
                    ));
                }
            }

            tensor_names.sort();
            files.push(WeightFile {
                relative_path,
                tensor_names,
            });
        }

        files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

        Ok(Self {
            root,
            format: SAFETENSORS_FILE_EXTENSION.to_string(),
            files,
            tensors,
        })
    }

    pub fn load_default() -> Result<Self, WeightError> {
        Self::from_root(DEFAULT_WEIGHT_ROOT)
    }

    pub fn tensor(&self, name: &str) -> Option<&WeightTensorDescriptor> {
        self.tensors.get(name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<LoadedTensor, WeightError> {
        let descriptor = self
            .tensor(name)
            .cloned()
            .ok_or_else(|| WeightError::TensorNotFound(name.to_string()))?;
        let absolute_path = self.root.join(&descriptor.relative_path);

        let mut file = File::open(&absolute_path).map_err(|source| WeightError::Io {
            path: absolute_path.clone(),
            source,
        })?;
        let header_length = read_header_length(&mut file, &absolute_path)?;
        let data_start = 8 + header_length as u64 + descriptor.data_offsets.0 as u64;
        let data_len = descriptor.data_offsets.1 - descriptor.data_offsets.0;

        file.seek(SeekFrom::Start(data_start))
            .map_err(|source| WeightError::Io {
                path: absolute_path.clone(),
                source,
            })?;
        let mut data = vec![0_u8; data_len];
        file.read_exact(&mut data)
            .map_err(|source| WeightError::Io {
                path: absolute_path,
                source,
            })?;

        Ok(LoadedTensor {
            name: descriptor.name,
            dtype: descriptor.dtype,
            shape: descriptor.shape,
            data,
        })
    }
}

fn discover_safetensors_files(root: &Path) -> Result<Vec<PathBuf>, WeightError> {
    let mut files = Vec::new();
    for entry in fs::read_dir(root).map_err(|source| WeightError::Io {
        path: root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| WeightError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_FILE_EXTENSION) {
            let relative_path = path
                .strip_prefix(root)
                .expect("discovered file must be under root")
                .to_path_buf();
            files.push(relative_path);
        }
    }

    files.sort();
    Ok(files)
}

fn read_safetensors_header(path: &Path) -> Result<Vec<(String, RawTensorHeader)>, WeightError> {
    let mut file = File::open(path).map_err(|source| WeightError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let header_length = read_header_length(&mut file, path)?;
    let mut header_bytes = vec![0_u8; header_length as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|source| WeightError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let header_map: Map<String, Value> =
        serde_json::from_slice(&header_bytes).map_err(|source| WeightError::InvalidHeaderJson {
            path: path.to_path_buf(),
            source,
        })?;

    let mut tensors = Vec::new();
    for (name, value) in header_map {
        if name == "__metadata__" {
            continue;
        }

        let header = RawTensorHeader::from_value(path, &name, value)?;
        tensors.push((name, header));
    }

    tensors.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(tensors)
}

fn read_header_length(file: &mut File, path: &Path) -> Result<u64, WeightError> {
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|source| WeightError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let header_length = u64::from_le_bytes(prefix);
    let total_length = file
        .metadata()
        .map_err(|source| WeightError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if 8 + header_length > total_length {
        return Err(WeightError::InvalidHeaderLength {
            path: path.to_path_buf(),
            header_length,
            file_length: total_length,
        });
    }

    Ok(header_length)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

impl RawTensorHeader {
    fn from_value(path: &Path, tensor_name: &str, value: Value) -> Result<Self, WeightError> {
        let value = value
            .as_object()
            .ok_or_else(|| WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "tensor entry must be an object".to_string(),
            })?;

        let dtype = value
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "missing dtype".to_string(),
            })?
            .to_string();

        let shape = value
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "missing shape".to_string(),
            })?
            .iter()
            .map(|value| {
                value.as_u64().map(|dim| dim as usize).ok_or_else(|| {
                    WeightError::InvalidTensorHeader {
                        path: path.to_path_buf(),
                        tensor: tensor_name.to_string(),
                        reason: "shape entries must be unsigned integers".to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let data_offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "missing data_offsets".to_string(),
            })?;
        if data_offsets.len() != 2 {
            return Err(WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "data_offsets must contain [start, end]".to_string(),
            });
        }

        let start = data_offsets[0]
            .as_u64()
            .ok_or_else(|| WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "data_offsets start must be an unsigned integer".to_string(),
            })? as usize;
        let end = data_offsets[1]
            .as_u64()
            .ok_or_else(|| WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "data_offsets end must be an unsigned integer".to_string(),
            })? as usize;

        if end < start {
            return Err(WeightError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: tensor_name.to_string(),
                reason: "data_offsets end must be greater than or equal to start".to_string(),
            });
        }

        Ok(Self {
            dtype,
            shape,
            data_offsets: (start, end),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct WeightIndexFile {
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

fn read_weight_index(path: &Path) -> Result<WeightIndexFile, WeightError> {
    let bytes = fs::read(path).map_err(|source| WeightError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| WeightError::InvalidIndexJson {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Debug)]
pub enum WeightError {
    RootNotFound(PathBuf),
    NoSafetensorsFiles(PathBuf),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidHeaderLength {
        path: PathBuf,
        header_length: u64,
        file_length: u64,
    },
    InvalidHeaderJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidIndexJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidTensorHeader {
        path: PathBuf,
        tensor: String,
        reason: String,
    },
    DuplicateTensorName(String),
    IndexMismatch {
        tensor: String,
        expected: PathBuf,
        actual: PathBuf,
    },
    TensorNotFound(String),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RootNotFound(path) => write!(f, "weight root does not exist: {}", path.display()),
            Self::NoSafetensorsFiles(path) => {
                write!(f, "no .safetensors files found under {}", path.display())
            }
            Self::Io { path, source } => {
                write!(f, "I/O error while reading {}: {}", path.display(), source)
            }
            Self::InvalidHeaderLength {
                path,
                header_length,
                file_length,
            } => write!(
                f,
                "invalid safetensors header length in {}: header={} file={}",
                path.display(),
                header_length,
                file_length
            ),
            Self::InvalidHeaderJson { path, source } => write!(
                f,
                "invalid safetensors header JSON in {}: {}",
                path.display(),
                source
            ),
            Self::InvalidIndexJson { path, source } => write!(
                f,
                "invalid safetensors index JSON in {}: {}",
                path.display(),
                source
            ),
            Self::InvalidTensorHeader {
                path,
                tensor,
                reason,
            } => write!(
                f,
                "invalid tensor header for {tensor} in {}: {reason}",
                path.display()
            ),
            Self::DuplicateTensorName(name) => {
                write!(f, "duplicate tensor name in manifest: {name}")
            }
            Self::IndexMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "index mismatch for tensor {tensor}: expected {}, got {}",
                expected.display(),
                actual.display()
            ),
            Self::TensorNotFound(name) => write!(f, "tensor not found: {name}"),
        }
    }
}

impl Error for WeightError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidHeaderJson { source, .. } => Some(source),
            Self::InvalidIndexJson { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("nemotron-weights-{name}-{suffix}"))
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

        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&data);

        fs::write(path, file_bytes).unwrap();
    }

    #[test]
    fn loads_single_safetensors_manifest_and_tensor() {
        let root = temp_dir("single");
        fs::create_dir_all(&root).unwrap();
        write_safetensors_file(
            &root.join("model.safetensors"),
            &[
                ("embed.weight", "F32", &[2], &[1, 2, 3, 4, 5, 6, 7, 8]),
                ("lm_head.weight", "U8", &[3], &[9, 10, 11]),
            ],
        );

        let manifest = WeightManifest::from_root(&root).unwrap();
        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.tensors.len(), 2);
        assert_eq!(
            manifest.tensor("embed.weight").unwrap(),
            &WeightTensorDescriptor {
                name: "embed.weight".to_string(),
                relative_path: PathBuf::from("model.safetensors"),
                dtype: "F32".to_string(),
                shape: vec![2],
                data_offsets: (0, 8),
            }
        );

        let tensor = manifest.load_tensor("lm_head.weight").unwrap();
        assert_eq!(tensor.dtype, "U8");
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.data, vec![9, 10, 11]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_indexed_shards() {
        let root = temp_dir("sharded");
        fs::create_dir_all(&root).unwrap();
        write_safetensors_file(
            &root.join("model-00001-of-00002.safetensors"),
            &[("layer0.weight", "U8", &[2], &[1, 2])],
        );
        write_safetensors_file(
            &root.join("model-00002-of-00002.safetensors"),
            &[("layer1.weight", "U8", &[2], &[3, 4])],
        );
        fs::write(
            root.join(SAFETENSORS_INDEX_FILE),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "layer0.weight": "model-00001-of-00002.safetensors",
                    "layer1.weight": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let manifest = WeightManifest::from_root(&root).unwrap();
        assert_eq!(manifest.files.len(), 2);
        assert_eq!(
            manifest.tensor("layer1.weight").unwrap().relative_path,
            PathBuf::from("model-00002-of-00002.safetensors")
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_root() {
        let error = WeightManifest::from_root("/definitely/missing/weights").unwrap_err();
        assert!(matches!(error, WeightError::RootNotFound(_)));
    }

    #[test]
    fn rejects_missing_tensor() {
        let root = temp_dir("missing-tensor");
        fs::create_dir_all(&root).unwrap();
        write_safetensors_file(&root.join("model.safetensors"), &[("x", "U8", &[1], &[1])]);

        let manifest = WeightManifest::from_root(&root).unwrap();
        let error = manifest.load_tensor("missing").unwrap_err();
        assert_eq!(error.to_string(), "tensor not found: missing");

        fs::remove_dir_all(root).unwrap();
    }
}
