"""Fixed inputs of the detector: concept text embeddings and per-step timestep embeddings."""

import torch

CONCEPTS = ["hate", "harassment", "violence", "self_harm", "sexual", "shocking", "illegal_activity"]
CONCEPT_TEMPLATE = "A photo of {}"


@torch.no_grad()
def concept_embeddings(tokenizer, text_encoder, concepts=CONCEPTS, template=CONCEPT_TEMPLATE):
    """CLIP text embedding of each concept, as {name: [1, 768]}.

    Uses the hidden state at the final (padding) position of the 77-token sequence.
    Released checkpoints were trained with this choice.
    """
    out = {}
    for concept in concepts:
        ids = tokenizer(
            template.format(concept.replace("_", " ")),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(text_encoder.device)
        hidden = text_encoder(ids)[0]
        out[concept] = hidden[0, -1:].float().cpu()
    return out


@torch.no_grad()
def timestep_embeddings(unet, scheduler, num_inference_steps=50):
    """UNet time embedding for each sampling step, as [T, 1280] in denoising order."""
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps.to(unet.device)
    t_emb = unet.time_proj(timesteps).to(unet.dtype)
    return unet.time_embedding(t_emb).float().cpu()


def stack_concepts(embeddings, concepts):
    """{name: [1, 768]} -> [K, 768] in the given order."""
    missing = [c for c in concepts if c not in embeddings]
    if missing:
        raise KeyError(f"Concepts {missing} not in embedding file (available: {sorted(embeddings)})")
    return torch.cat([embeddings[c].reshape(1, -1) for c in concepts])
