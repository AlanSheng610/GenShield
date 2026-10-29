import torch

from .embeddings import stack_concepts
from .model import C3A


class GenShield:
    """Scores bottleneck latents against a set of unsafe concepts.

    Args:
        model: a trained :class:`C3A`.
        concept_embeddings: {name: [1, 768]} CLIP text embeddings.
        timestep_table: [T, 1280] timestep embedding for each sampling step.
        concepts: which concepts to score (default: all in ``concept_embeddings``).
        threshold: similarity above which a latent is flagged as unsafe.
    """

    def __init__(self, model, concept_embeddings, timestep_table, concepts=None, threshold=0.87):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.concepts = list(concepts or concept_embeddings)
        self.concept_matrix = stack_concepts(concept_embeddings, self.concepts).to(self.device)
        self.timestep_table = timestep_table.to(self.device)
        self.threshold = threshold

    @classmethod
    def from_files(cls, checkpoint, concepts_path, timesteps_path, concepts=None,
                   nhead=8, threshold=0.87, device="cuda"):
        model = C3A(nhead=nhead)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return cls(
            model.to(device),
            torch.load(concepts_path, map_location="cpu", weights_only=True),
            torch.load(timesteps_path, map_location="cpu", weights_only=True),
            concepts=concepts,
            threshold=threshold,
        )

    @torch.no_grad()
    def score(self, latents, steps):
        """Similarity to every concept.

        Args:
            latents: [B, 1280, 8, 8] or [B, 1280, 64].
            steps: sampling-step index (int) or [B] tensor of indices.

        Returns:
            [B, K] cosine similarities.
        """
        latents = latents.to(self.device).flatten(2)
        if isinstance(steps, int):
            steps = torch.full((latents.shape[0],), steps, device=self.device)
        return self.model(latents, self.timestep_table[steps.to(self.device)], self.concept_matrix)

    @torch.no_grad()
    def score_sequence(self, latents):
        """[T, 1280, ...] latents of one generation -> [T, K] similarities."""
        return self.score(latents, torch.arange(latents.shape[0], device=self.device))
