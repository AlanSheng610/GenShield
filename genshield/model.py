"""Concept-Content Cross-Attention (C3A).

Concept embeddings (CLIP text features) act as queries; diffusion latents from the
UNet bottleneck, modulated by the timestep embedding through FiLM, act as keys and
values. Both sides are projected into a shared 128-d space where the cosine
similarity between a latent and a concept measures how strongly that concept is
manifesting in the image being generated.

Attribute names (``cross_attn``, ``film``, ``mlp_concept``, ``mlp_cross``) and the
layer order of the projection heads are part of the checkpoint format; changing them
breaks loading of released weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_DIM = 768        # SD-1.5 text-encoder width
LATENT_DIM = 1280     # channels of the SD-1.5 UNet mid-block
TIMESTEP_DIM = 1280   # width of the SD-1.5 UNet time embedding
PROJ_DIM = 128


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, query_dim=CLIP_DIM):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.W_q = nn.Linear(query_dim, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)

    def forward(self, q, kv):
        """q: [B, Nq, query_dim], kv: [B, Nkv, d_model] -> [B, Nq, d_model]."""
        out = F.scaled_dot_product_attention(
            self._split_heads(self.W_q(q)),
            self._split_heads(self.W_k(kv)),
            self._split_heads(self.W_v(kv)),
        )
        b, _, n, _ = out.shape
        return self.W_o(out.transpose(1, 2).reshape(b, n, -1))


def _projection_head(in_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(256, PROJ_DIM),
    )


class C3A(nn.Module):
    def __init__(self, d_model=LATENT_DIM, nhead=8, dropout=0.1, timestep_dim=TIMESTEP_DIM):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, nhead)
        self.film = nn.Linear(timestep_dim, d_model * 2)
        self.mlp_concept = _projection_head(CLIP_DIM, dropout)
        self.mlp_cross = _projection_head(d_model, dropout)

    def film_modulate(self, latents, timestep_emb):
        """latents: [B, C, N] channel-first tokens; timestep_emb: [B, T_dim] or [T_dim]."""
        x = latents.float().transpose(1, 2)  # [B, N, C]
        scale, shift = self.film(timestep_emb.float()).chunk(2, dim=-1)
        if scale.dim() == 1:
            scale, shift = scale.unsqueeze(0), shift.unsqueeze(0)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def encode_content(self, latents, timestep_emb, concepts):
        """Embed latents with respect to each concept.

        Args:
            latents: [B, C, N] bottleneck features (an 8x8 map flattened to N=64).
            timestep_emb: [B, T_dim] or [T_dim].
            concepts: [K, CLIP_DIM] shared by the batch, or [B, K, CLIP_DIM].

        Returns:
            [B, K, PROJ_DIM]. Queries attend independently, so the output for one
            concept does not depend on which other concepts are in the batch.
        """
        kv = self.film_modulate(latents, timestep_emb)
        if concepts.dim() == 2:
            concepts = concepts.unsqueeze(0).expand(kv.shape[0], -1, -1)
        return self.mlp_cross(self.cross_attn(concepts.float(), kv))

    def encode_concepts(self, concepts):
        """[K, CLIP_DIM] -> [K, PROJ_DIM]."""
        return self.mlp_concept(concepts.float())

    def forward(self, latents, timestep_emb, concepts):
        """Cosine similarity [B, K] between every latent and every concept."""
        z = self.encode_content(latents, timestep_emb, concepts)
        z_c = self.encode_concepts(concepts)
        return F.cosine_similarity(z, z_c.unsqueeze(0), dim=-1)

    def contrastive_features(self, latents, timestep_emb, labels, concepts):
        """Build the feature set for safety-aware contrastive learning.

        Unsafe latents (label k > 0) attend only to their own concept k. Safe latents
        (label 0) attend to every concept, and each of those K outputs is a negative.
        The projected concept embeddings are appended with labels 1..K as anchors.

        Returns:
            features [M, PROJ_DIM] and labels [M].
        """
        features, out_labels = [], []

        unsafe = labels > 0
        if unsafe.any():
            own_concept = concepts[labels[unsafe] - 1].unsqueeze(1)  # [Nu, 1, CLIP_DIM]
            z = self.encode_content(latents[unsafe], timestep_emb[unsafe], own_concept)
            features.append(z.squeeze(1))
            out_labels.append(labels[unsafe])

        safe = ~unsafe
        if safe.any():
            z = self.encode_content(latents[safe], timestep_emb[safe], concepts)
            features.append(z.flatten(0, 1))
            out_labels.append(labels.new_zeros(z.shape[0] * z.shape[1]))

        features.append(self.encode_concepts(concepts))
        out_labels.append(torch.arange(1, len(concepts) + 1, device=labels.device))

        return torch.cat(features), torch.cat(out_labels)
