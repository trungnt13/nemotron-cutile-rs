use crate::{LayerStub, Mamba2Cache};
use std::error::Error;
use std::fmt;

pub const SPEC: LayerStub = LayerStub {
    name: "cache",
    summary: "Attention KV-cache and hybrid per-layer cache containers.",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheShape {
    pub batch_size: usize,
    pub key_value_head_count: usize,
    pub head_dim: usize,
}

impl KvCacheShape {
    pub const fn new(batch_size: usize, key_value_head_count: usize, head_dim: usize) -> Self {
        Self {
            batch_size,
            key_value_head_count,
            head_dim,
        }
    }

    pub const fn width(self) -> usize {
        self.key_value_head_count * self.head_dim
    }

    pub const fn row_count(self, sequence_len: usize) -> usize {
        self.batch_size * sequence_len
    }

    pub const fn len_for_sequence(self, sequence_len: usize) -> usize {
        self.row_count(sequence_len) * self.width()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvCache {
    shape: KvCacheShape,
    sequence_len: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvCache {
    pub fn new(shape: KvCacheShape) -> Result<Self, CacheError> {
        validate_kv_shape(shape)?;
        Ok(Self {
            shape,
            sequence_len: 0,
            keys: Vec::new(),
            values: Vec::new(),
        })
    }

    pub const fn shape(&self) -> KvCacheShape {
        self.shape
    }

    pub const fn sequence_len(&self) -> usize {
        self.sequence_len
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        appended_sequence_len: usize,
    ) -> Result<(), CacheError> {
        if appended_sequence_len == 0 {
            return Err(CacheError::InvalidAppendLength(appended_sequence_len));
        }

        let expected = self.shape.len_for_sequence(appended_sequence_len);
        if keys.len() != expected {
            return Err(CacheError::LengthMismatch {
                argument: "keys",
                expected,
                actual: keys.len(),
            });
        }

        if values.len() != expected {
            return Err(CacheError::LengthMismatch {
                argument: "values",
                expected,
                actual: values.len(),
            });
        }

        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        self.sequence_len += appended_sequence_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.sequence_len = 0;
        self.keys.clear();
        self.values.clear();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerCache {
    Empty,
    Attention(KvCache),
    Mamba(Mamba2Cache),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HybridCache {
    layers: Vec<LayerCache>,
}

impl HybridCache {
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![LayerCache::Empty; layer_count],
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layer(&self, layer_index: usize) -> Result<&LayerCache, CacheError> {
        self.layers
            .get(layer_index)
            .ok_or(CacheError::LayerOutOfBounds {
                layer_index,
                layer_count: self.layers.len(),
            })
    }

    pub fn layer_mut(&mut self, layer_index: usize) -> Result<&mut LayerCache, CacheError> {
        let layer_count = self.layers.len();
        self.layers
            .get_mut(layer_index)
            .ok_or(CacheError::LayerOutOfBounds {
                layer_index,
                layer_count,
            })
    }

    pub fn set_attention(
        &mut self,
        layer_index: usize,
        cache: KvCache,
    ) -> Result<(), CacheError> {
        *self.layer_mut(layer_index)? = LayerCache::Attention(cache);
        Ok(())
    }

    pub fn set_mamba(
        &mut self,
        layer_index: usize,
        cache: Mamba2Cache,
    ) -> Result<(), CacheError> {
        *self.layer_mut(layer_index)? = LayerCache::Mamba(cache);
        Ok(())
    }

    pub fn clear_layer(&mut self, layer_index: usize) -> Result<(), CacheError> {
        *self.layer_mut(layer_index)? = LayerCache::Empty;
        Ok(())
    }

    pub fn clear_all(&mut self) {
        for layer in &mut self.layers {
            *layer = LayerCache::Empty;
        }
    }
}

fn validate_kv_shape(shape: KvCacheShape) -> Result<(), CacheError> {
    if shape.batch_size == 0 || shape.key_value_head_count == 0 || shape.head_dim == 0 {
        return Err(CacheError::InvalidKvShape(shape));
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum CacheError {
    InvalidKvShape(KvCacheShape),
    InvalidAppendLength(usize),
    LengthMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
    LayerOutOfBounds {
        layer_index: usize,
        layer_count: usize,
    },
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidKvShape(shape) => write!(
                f,
                "kv cache shape must be non-zero, got batch_size={}, key_value_head_count={}, head_dim={}",
                shape.batch_size, shape.key_value_head_count, shape.head_dim
            ),
            Self::InvalidAppendLength(sequence_len) => {
                write!(f, "appended sequence length must be non-zero, got {sequence_len}")
            }
            Self::LengthMismatch {
                argument,
                expected,
                actual,
            } => write!(
                f,
                "{argument} length mismatch: expected {expected}, got {actual}"
            ),
            Self::LayerOutOfBounds {
                layer_index,
                layer_count,
            } => write!(
                f,
                "layer index {layer_index} is out of bounds for cache with {layer_count} layers"
            ),
        }
    }
}

impl Error for CacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_key_value_rows() {
        let mut cache = KvCache::new(KvCacheShape::new(1, 1, 2)).unwrap();

        cache.append(&[1.0, 2.0], &[3.0, 4.0], 1).unwrap();
        cache.append(&[5.0, 6.0], &[7.0, 8.0], 1).unwrap();

        assert_eq!(cache.sequence_len(), 2);
        assert_eq!(cache.keys(), &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(cache.values(), &[3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn clear_resets_cache() {
        let mut cache = KvCache::new(KvCacheShape::new(1, 1, 1)).unwrap();
        cache.append(&[1.0], &[2.0], 1).unwrap();
        cache.clear();

        assert_eq!(cache.sequence_len(), 0);
        assert!(cache.keys().is_empty());
        assert!(cache.values().is_empty());
    }

    #[test]
    fn hybrid_cache_stores_attention_and_mamba_entries() {
        let mut hybrid = HybridCache::new(2);
        let attention = KvCache::new(KvCacheShape::new(1, 1, 1)).unwrap();
        let mamba = Mamba2Cache::new_zeroed(1, 2, 4, 3);

        hybrid.set_attention(0, attention.clone()).unwrap();
        hybrid.set_mamba(1, mamba.clone()).unwrap();

        assert_eq!(hybrid.layer(0).unwrap(), &LayerCache::Attention(attention));
        assert_eq!(hybrid.layer(1).unwrap(), &LayerCache::Mamba(mamba));
    }

    #[test]
    fn rejects_length_mismatch_on_append() {
        let mut cache = KvCache::new(KvCacheShape::new(1, 1, 2)).unwrap();
        let error = cache.append(&[1.0], &[2.0, 3.0], 1).unwrap_err();

        assert_eq!(
            error,
            CacheError::LengthMismatch {
                argument: "keys",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_invalid_layer_index() {
        let hybrid = HybridCache::new(1);
        let error = hybrid.layer(5).unwrap_err();

        assert_eq!(
            error,
            CacheError::LayerOutOfBounds {
                layer_index: 5,
                layer_count: 1,
            }
        );
    }
}
