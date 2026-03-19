use candle_core::Tensor;
use candle_nn::{embedding, layer_norm, linear, Activation, Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use anyhow::Result;

use crate::config::KittenConfig;

struct AlbertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl AlbertEmbeddings {
    fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let vb_e = vb.pp("embeddings");
        let word_embeddings = embedding(cfg.n_token - 1, cfg.bert_embed_dim, vb_e.pp("word_embeddings"))?;
        let token_type_embeddings = embedding(2, cfg.bert_embed_dim, vb_e.pp("token_type_embeddings"))?;
        let position_embeddings = embedding(cfg.bert_max_pos, cfg.bert_embed_dim, vb_e.pp("position_embeddings"))?;
        let layer_norm = layer_norm(cfg.bert_embed_dim, LayerNormConfig::default(), vb_e.pp("LayerNorm"))?;
        Ok(Self { word_embeddings, token_type_embeddings, position_embeddings, layer_norm })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let dev = input_ids.device();

        let word_emb = self.word_embeddings.forward(input_ids)?;

        // token type ids = all zeros
        let type_ids = Tensor::zeros((1, seq_len), candle_core::DType::U32, dev)?;
        let type_emb = self.token_type_embeddings.forward(&type_ids)?;

        // position ids = [0, 1, ..., seq_len-1]
        let pos_ids = Tensor::arange(0u32, seq_len as u32, dev)?.unsqueeze(0)?;
        let pos_emb = self.position_embeddings.forward(&pos_ids)?;

        let emb = (word_emb + type_emb)?.add(&pos_emb)?;
        Ok(self.layer_norm.forward(&emb)?)
    }
}

struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    n_heads: usize,
    head_dim: usize,
}

impl AlbertAttention {
    fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let vb_a = vb.pp("attention");
        let n_heads = cfg.bert_n_heads;
        let head_dim = cfg.bert_hidden_dim / n_heads;
        let query = linear(cfg.bert_hidden_dim, cfg.bert_hidden_dim, vb_a.pp("query"))?;
        let key = linear(cfg.bert_hidden_dim, cfg.bert_hidden_dim, vb_a.pp("key"))?;
        let value = linear(cfg.bert_hidden_dim, cfg.bert_hidden_dim, vb_a.pp("value"))?;
        let dense = linear(cfg.bert_hidden_dim, cfg.bert_hidden_dim, vb_a.pp("dense"))?;
        let layer_norm = layer_norm(cfg.bert_hidden_dim, LayerNormConfig::default(), vb_a.pp("LayerNorm"))?;
        Ok(Self { query, key, value, dense, layer_norm, n_heads, head_dim })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = hidden.dims3()?;
        let scale = (self.head_dim as f64).sqrt();

        let project = |linear: &Linear| -> Result<Tensor> {
            let x = linear.forward(hidden)?;
            // [batch, seq, hidden] -> [batch, n_heads, seq, head_dim]
            Ok(x.reshape((batch, seq, self.n_heads, self.head_dim))?
                .transpose(1, 2)?)
        };

        let q = project(&self.query)?;
        let k = project(&self.key)?;
        let v = project(&self.value)?;

        // [batch, n_heads, seq, seq]
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn = candle_nn::ops::softmax(&attn, 3)?;

        // [batch, n_heads, seq, head_dim] -> [batch, seq, hidden]
        let ctx = attn.matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, self.n_heads * self.head_dim))?;

        let out = self.dense.forward(&ctx)?;
        Ok(self.layer_norm.forward(&(out + hidden)?)?)
    }
}

struct AlbertLayer {
    attention: AlbertAttention,
    ffn: Linear,
    ffn_output: Linear,
    full_layer_layer_norm: LayerNorm,
}

impl AlbertLayer {
    fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let attention = AlbertAttention::load(vb.clone(), cfg)?;
        let ffn = linear(cfg.bert_hidden_dim, cfg.bert_ffn_dim, vb.pp("ffn"))?;
        let ffn_output = linear(cfg.bert_ffn_dim, cfg.bert_hidden_dim, vb.pp("ffn_output"))?;
        let full_layer_layer_norm = layer_norm(cfg.bert_hidden_dim, LayerNormConfig::default(), vb.pp("full_layer_layer_norm"))?;
        Ok(Self { attention, ffn, ffn_output, full_layer_layer_norm })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let attn_out = self.attention.forward(hidden)?;
        let ffn_mid = Activation::Gelu.forward(&self.ffn.forward(&attn_out)?)?;
        let ffn_out = self.ffn_output.forward(&ffn_mid)?;
        Ok(self.full_layer_layer_norm.forward(&(ffn_out + &attn_out)?)?)
    }
}

pub struct AlbertEncoder {
    embeddings: AlbertEmbeddings,
    embedding_hidden_mapping_in: Linear,
    shared_layer: AlbertLayer,
    n_layers: usize,
    bert_encoder: Linear,
}

impl AlbertEncoder {
    pub fn load(vb: VarBuilder, cfg: &KittenConfig) -> Result<Self> {
        let vb_bert = vb.pp("bert");
        let embeddings = AlbertEmbeddings::load(vb_bert.clone(), cfg)?;
        let vb_enc = vb_bert.pp("encoder");
        let embedding_hidden_mapping_in = linear(cfg.bert_embed_dim, cfg.bert_hidden_dim, vb_enc.pp("embedding_hidden_mapping_in"))?;
        let shared_layer = AlbertLayer::load(
            vb_enc.pp("albert_layer_groups").pp("0").pp("albert_layers").pp("0"),
            cfg,
        )?;
        let bert_encoder = linear(cfg.bert_hidden_dim, cfg.bert_embed_dim, vb.pp("bert_encoder"))?;
        Ok(Self {
            embeddings,
            embedding_hidden_mapping_in,
            shared_layer,
            n_layers: 12,
            bert_encoder,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let emb = self.embeddings.forward(input_ids)?;
        let mut hidden = self.embedding_hidden_mapping_in.forward(&emb)?;
        for _ in 0..self.n_layers {
            hidden = self.shared_layer.forward(&hidden)?;
        }
        Ok(self.bert_encoder.forward(&hidden)?)
    }
}
