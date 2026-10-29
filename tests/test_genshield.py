import numpy as np
import pytest
import torch

from genshield.data import LatentSequenceDataset, LatentStepDataset, Source, load_latent_file
from genshield.detector import GenShield
from genshield.hooks import LatentRecorder
from genshield.losses import SupConLoss
from genshield.metrics import accuracy_over_steps, balanced_subsample, detection_metrics, fpr_at_tpr
from genshield.model import C3A

K = 3


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return C3A(nhead=8).eval()


def test_forward_shapes(model):
    sim = model(torch.randn(4, 1280, 64), torch.randn(4, 1280), torch.randn(K, 768))
    assert sim.shape == (4, K)
    assert sim.abs().max() <= 1 + 1e-6


def test_concepts_attend_independently(model):
    x, t, c = torch.randn(2, 1280, 64), torch.randn(2, 1280), torch.randn(K, 768)
    with torch.no_grad():
        joint = model(x, t, c)
        single = torch.cat([model(x, t, c[k:k + 1]) for k in range(K)], dim=1)
    assert torch.allclose(joint, single, atol=1e-5)


def test_contrastive_features(model):
    labels = torch.tensor([0, 1, 3, 0, 2])
    feats, out = model.contrastive_features(torch.randn(5, 1280, 64), torch.randn(5, 1280), labels, torch.randn(K, 768))
    # 3 unsafe (own concept) + 2 safe x K concepts + K anchors
    assert feats.shape == (3 + 2 * K + K, 128)
    assert out.tolist() == [1, 3, 2] + [0] * (2 * K) + [1, 2, 3]


def test_supcon_prefers_clustered_features():
    labels = torch.tensor([0, 0, 1, 1])
    clustered = torch.tensor([[1.0, 0], [1, 0], [0, 1], [0, 1]])
    mixed = torch.tensor([[1.0, 0], [0, 1], [1, 0], [0, 1]])
    loss = SupConLoss()
    assert loss(clustered, labels) < loss(mixed, labels)


def test_metrics():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = detection_metrics(labels, scores, threshold=0.5)
    assert m["auroc"] == 1.0 and m["fpr@tpr95"] == 0.0 and m["f1"] == 1.0
    assert fpr_at_tpr(labels, 1 - scores) == 1.0

    step_scores = np.array([[0.1, 0.9], [0.95, 0.2]])
    assert accuracy_over_steps(step_scores, np.array([1, 0]), 0.5).tolist() == [0.0, 0.5]

    safe, unsafe = balanced_subsample(np.arange(10), np.arange(3))
    assert len(safe) == len(unsafe) == 3


def test_latent_files(tmp_path):
    t, c = 4, 1280
    (tmp_path / "safe").mkdir()
    (tmp_path / "unsafe").mkdir()
    # Current format, plus both layouts written by the original research code.
    torch.save({"latents": torch.randn(t, c, 8, 8).half(), "prompt": "a cat"}, tmp_path / "safe" / "latents_0.pt")
    torch.save(torch.randn(t, c, 8, 8), tmp_path / "unsafe" / "bottleneck_features_10.pt")
    torch.save({"bottleneck_features": torch.randn(t, 1, c, 8, 8), "prompt": "x"},
               tmp_path / "unsafe" / "bottleneck_features_2.pt")

    latents, prompt = load_latent_file(tmp_path / "safe" / "latents_0.pt")
    assert latents.shape == (t, c, 64) and latents.dtype == torch.float32 and prompt == "a cat"

    sources = [Source(str(tmp_path / "safe"), 0), Source(str(tmp_path / "unsafe"), 2)]
    seq = LatentSequenceDataset(sources)
    assert [label for _, label in seq.items] == [0, 2, 2]
    assert seq.items[1][0].endswith("_2.pt")  # numeric, not lexicographic, order

    steps = LatentStepDataset(sources, steps=range(t))
    assert len(steps) == 3 * t
    x, step, label = steps[t + 1]
    assert x.shape == (c, 64) and step == 1 and label == 2


def test_detector_scores_every_step(model):
    concepts = {f"c{k}": torch.randn(1, 768) for k in range(K)}
    shield = GenShield(model, concepts, torch.randn(50, 1280), threshold=0.5)
    assert shield.score(torch.randn(2, 1280, 8, 8), 7).shape == (2, K)
    assert shield.score_sequence(torch.randn(50, 1280, 8, 8)).shape == (50, K)


def test_latent_recorder_keeps_text_branch():
    diffusers = pytest.importorskip("diffusers")
    unet = diffusers.UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32, attention_head_dim=8,
        norm_num_groups=8,
    ).eval()
    sample = torch.randn(2, 4, 16, 16)
    text = torch.randn(2, 7, 32)

    with torch.no_grad(), LatentRecorder(unet, classifier_free_guidance=True) as cfg_rec:
        unet(sample, 10, encoder_hidden_states=text)
        unet(sample, 20, encoder_hidden_states=text)
    assert cfg_rec.stack().shape == (2, 1, 64, 8, 8)

    # The kept half is the text-conditioned one, i.e. the second element of the CFG batch.
    with torch.no_grad(), LatentRecorder(unet, classifier_free_guidance=False) as rec:
        unet(sample[1:], 10, encoder_hidden_states=text[1:])
    assert torch.allclose(cfg_rec.latents[0], rec.latents[0], atol=1e-5)
    assert not unet.mid_block._forward_hooks  # hook removed on exit
