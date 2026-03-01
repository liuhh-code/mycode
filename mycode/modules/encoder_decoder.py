from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel
from .mamba_ssm.modules.mamba_simple import Mamba as Encoder


class GuidanceMemoryGenerator(nn.Module):
    def __init__(self, d_model, nhead, gm_rows=3):
        super(GuidanceMemoryGenerator, self).__init__()
        self.d_model = d_model
        self.gm_rows = gm_rows

        self.Vq = nn.Linear(d_model, d_model)
        self.Vk = nn.Linear(d_model, d_model)
        self.Vv = nn.Linear(d_model, d_model)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        self.mlp_res = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        self.Vf = nn.Linear(d_model, d_model)
        self.Jf = nn.Linear(d_model, d_model)
        self.Vi = nn.Linear(d_model, d_model)
        self.Ji = nn.Linear(d_model, d_model)

    def forward(self, gm_prev, y_prev):
        query = self.Vq(gm_prev)

        kv_input = torch.cat((gm_prev, y_prev), dim=1)
        key = self.Vk(kv_input)
        value = self.Vv(kv_input)

        O, _ = self.multihead_attn(query, key, value)

        gm_star = self.mlp_res(O + gm_prev) + O + gm_prev

        y_prev_expanded = y_prev.repeat(1, self.gm_rows, 1)

        forget_gate = torch.sigmoid(self.Vf(y_prev_expanded) + torch.tanh(self.Jf(gm_prev)))
        input_gate = torch.sigmoid(self.Vi(y_prev_expanded) + torch.tanh(self.Ji(gm_prev)))

        forget_gate = torch.clamp(forget_gate, 0.0, 0.6)
        input_gate = torch.clamp(input_gate, 0.0, 0.5)

        gm_star_scaled = torch.tanh(gm_star * 0.25)
        gm_t = gm_prev + 0.1 * (forget_gate * gm_prev + input_gate * gm_star_scaled - gm_prev)

        with torch.no_grad():
            gm_diff = torch.norm(gm_t - gm_prev, dim=-1).mean()
        return gm_t

class ContextGuidanceNormalizationLayer(nn.Module):
    def __init__(self, d_model):
        super(ContextGuidanceNormalizationLayer, self).__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        self.mlp_gamma = nn.Linear(d_model, d_model)
        self.mlp_beta = nn.Linear(d_model, d_model)

        self.eps = 1e-5

    def forward(self, x, gm_t):
        batch_size = x.size(0)
        gm_t_vector = gm_t.mean(dim=1)

        g_t = (self.gamma + 0.2 * self.mlp_gamma(gm_t_vector)).view(batch_size, 1, -1)
        b_t = (self.beta + 0.2 * self.mlp_beta(gm_t_vector)).view(batch_size, 1, -1)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return g_t * ((x - mean) / (std + self.eps)) + b_t


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        if mask.size(-1) != scores.size(-1):
            mask = query.new_ones((query.size(0), 1, 1, scores.size(-1)), dtype=torch.long)
            
        scores = scores.masked_fill(mask == 0, -1e9)
        
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm, gm_rows):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm
        self.initial_gm = nn.Parameter(torch.zeros(1, gm_rows, self.decoder.d_model))

    def init_state(self, batch_size, bos_idx, device):
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        rm_memory_state = self.rm.init_memory(batch_size).to(device)

        gm_state = self.initial_gm.expand(batch_size, -1, -1).to(device)

        return [
            ys.unsqueeze(0),
            rm_memory_state.unsqueeze(0),
            gm_state.unsqueeze(0)
        ]

    def forward(self, src, tgt, src_mask, tgt_mask, biomed_repr=None):
        hidden_states = self.encode(src, src_mask)
        tgt_embedded = self.tgt_embed(tgt)

        memory = self.rm.init_memory(tgt_embedded.size(0)).to(tgt_embedded.device)
        memory = self.rm(tgt_embedded, memory)

        gm_state = self.initial_gm.expand(tgt_embedded.size(0), -1, -1).to(tgt_embedded.device)

        return self.decode(hidden_states, src_mask, tgt_embedded, tgt_mask, memory, gm_state, biomed_repr=biomed_repr)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src))

    def decode(self, hidden_states, src_mask, tgt_embedded, tgt_mask, memory, gm_state, biomed_repr=None):
        return self.decoder(tgt_embedded, hidden_states, src_mask, tgt_mask, memory, gm_state, biomed_repr)

class Encoder_ori(nn.Module):
    def __init__(self, layer, N):
        super(Encoder_ori, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N, gm_rows):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = ContextGuidanceNormalizationLayer(layer.d_model)
        self.d_model = layer.d_model
        self.gm_rows = gm_rows

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, gm_state, biomed_repr=None):
        updated_gm_state = gm_state
        
        for layer in self.layers:
            x, updated_gm_state = layer(x, hidden_states, src_mask, tgt_mask, memory, updated_gm_state, biomed_repr)
        return self.norm(x, updated_gm_state), updated_gm_state


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model, gm_rows):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnectionWithCGM(d_model, dropout, gm_rows), 3)

        self.biomed_dropout = nn.Dropout(p=0.1)
        self.biomed_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        nn.init.constant_(self.biomed_gate[0].bias, 1.0)

        self.guidance_memory_generator = GuidanceMemoryGenerator(d_model, self_attn.h, gm_rows)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory, gm_prev, biomed_repr=None):
        m = memory 

        if src_mask is not None and src_mask.size(-1) != m.size(1):
            src_mask = m.new_ones((m.size(0), 1, 1, m.size(1)), dtype=torch.long)

        y_prev_for_gm_gen = x.mean(dim=1, keepdim=True)

        gm_t = self.guidance_memory_generator(gm_prev, y_prev_for_gm_gen)

        x = self.sublayer[0](x, lambda x_in: self.self_attn(x_in, x_in, x_in, tgt_mask), gm_t)
        x = self.sublayer[1](x, lambda x_in: self.src_attn(x_in, m, m, src_mask), gm_t)
        x = self.sublayer[2](x, self.feed_forward, gm_t)

        if biomed_repr is not None:
            biomed_repr = self.biomed_dropout(biomed_repr)
            gate = self.biomed_gate(torch.cat([x, biomed_repr], dim=-1))

            x = gate * x + (1 - gate) * biomed_repr

        return x, gm_t


class SublayerConnectionWithCGM(nn.Module):
    def __init__(self, d_model, dropout, gm_rows):
        super(SublayerConnectionWithCGM, self).__init__()
        self.norm = ContextGuidanceNormalizationLayer(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer, gm_t):
        ref_shape = x.shape
        
        normed_x = self.norm(x, gm_t)
        sub_out = sublayer(normed_x)
        
        if sub_out.shape != ref_shape:
            sub_out = sub_out.view(ref_shape)
            
        return x + self.dropout(sub_out)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model 
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        q_seq_len = query.size(1)
        k_seq_len = key.size(1)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        query = self.linears[0](query).view(nbatches, q_seq_len, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, k_seq_len, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, k_seq_len, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, q_seq_len, self.h * self.d_k)
        
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())
        self.memory_slots = nn.Parameter(torch.randn(num_slots, d_model))

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        return self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)

    def forward_step(self, input, memory_flat):
        memory = memory_flat.reshape(-1, self.num_slots, self.d_model)

        q = memory
        expanded_input = input.unsqueeze(1)

        k = torch.cat([memory, expanded_input], 1)
        v = torch.cat([memory, expanded_input], 1)

        next_memory_attn = memory + self.attn(q, k, v)
        next_memory_mlp = next_memory_attn + self.mlp(next_memory_attn)

        gates = self.W(expanded_input) + self.U(torch.tanh(memory))

        input_gate, forget_gate = torch.split(gates, split_size_or_sections=self.d_model, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory_mlp) + forget_gate * memory
        next_memory_flat = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory_flat

    def forward(self, inputs, memory):
        current_memory_flat = memory.view(memory.size(0), -1)

        for i in range(inputs.shape[1]):
            current_token_input = inputs[:, i, :]
            current_memory_flat = self.forward_step(current_token_input, current_memory_flat)

        final_memory = current_memory_flat.reshape(memory.size(0), self.num_slots, self.d_model)
        return final_memory

class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab, gm_rows):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
            Encoder(
                d_model=self.d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model, gm_rows),
                self.num_layers,
                gm_rows
            ),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm,
            gm_rows
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model
        self.gm_rows = getattr(args, 'gm_rows', 3)

        self.use_biomedbert = args.use_biomedbert
        if self.use_biomedbert:
            try:
                from modules.biomedbert_projector import BiomedBERTProjector
                self.biomedbert = BiomedBERTProjector(d_model=args.d_model)
            except ImportError:
                print("Warning: BiomedBERTProjector not found, proceeding without it.")
                self.biomedbert = None
                self.use_biomedbert = False

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab, self.gm_rows)

        self.core = self.model

        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        bos_idx = self.bos_idx
        device = self.args.device if hasattr(self.args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model.init_state(bsz, bos_idx, device)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, None)
        
        mamba_len = memory.size(1) 
        
        current_mask_len = att_masks.size(-1) if att_masks is not None else 0
        
        if mamba_len != current_mask_len and att_masks is not None:
            print(f"[Debug] Mask Length Mismatch! Mamba Length: {mamba_len}, Current Mask Length: {current_mask_len}")
            
            batch_size = memory.size(0)
            
            new_att_masks = memory.new_ones((batch_size, 1, mamba_len), dtype=torch.bool)
            
            att_masks = new_att_masks
            print(f"[Debug] att_masks fixed. New Shape: {att_masks.shape}")
        
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        biomed_repr = None
        if self.use_biomedbert and self.biomedbert is not None:
            biomed_ids = seq
            biomed_mask = (biomed_ids > 0).long()
            biomed_repr = self.biomedbert(biomed_ids, biomed_mask)

        if biomed_repr is not None:
            biomed_repr = None

        out, _ = self.model(
            src=att_feats,
            tgt=seq,
            src_mask=att_masks,
            tgt_mask=seq_mask,
            biomed_repr=biomed_repr
        )
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, encoder_out, state, att_masks):

        if len(state) == 0:
            ys = it.unsqueeze(1)
            rm_memory_state = self.model.rm.init_memory(encoder_out.size(0)).to(encoder_out.device)
            gm_state = self.model.initial_gm.expand(encoder_out.size(0), -1, -1).to(encoder_out.device)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

            rm_memory_state = state[1][0]
            gm_state = state[2][0]

        biomed_repr = None
        if self.use_biomedbert and self.biomedbert is not None:
            biomed_ids = ys
            biomed_mask = (biomed_ids > 0).long()
            biomed_repr = self.biomedbert(biomed_ids, biomed_mask)
            
        if biomed_repr is not None:
            biomed_repr = None

        tgt_embedded = self.model.tgt_embed(ys)

        current_token_embedded = tgt_embedded[:, -1, :]
        rm_memory_state_flat = rm_memory_state.view(rm_memory_state.size(0), -1)
        rm_memory_state_updated_flat = self.model.rm.forward_step(current_token_embedded, rm_memory_state_flat)
        rm_memory_state = rm_memory_state_updated_flat.reshape(rm_memory_state_updated_flat.size(0), self.rm_num_slots, self.rm_d_model)

        tgt_mask = subsequent_mask(ys.size(1)).to(encoder_out.device)

        out, gm_state_updated = self.model.decode(encoder_out, att_masks, tgt_embedded, tgt_mask, rm_memory_state,
                                                  gm_state, biomed_repr=biomed_repr)

        return out[:, -1], [ys.unsqueeze(0), rm_memory_state.unsqueeze(0),
                            gm_state_updated.unsqueeze(0)]


if __name__ == '__main__':
    model = Encoder(
        d_model=512,
        d_state=16,
        d_conv=4,
        expand=2,
    )
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable Parameters: {params}")