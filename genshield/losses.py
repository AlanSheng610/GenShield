import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020, https://arxiv.org/abs/2004.11362).

    Single-view variant: every feature is an anchor, and features sharing its label
    are its positives.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """features: [N, D], labels: [N]."""
        n = features.shape[0]
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float()

        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # numerical stability

        not_self = 1.0 - torch.eye(n, device=features.device)
        positive_mask = positive_mask * not_self

        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Anchors without any positive contribute zero instead of NaN.
        num_positives = positive_mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / num_positives

        return -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
