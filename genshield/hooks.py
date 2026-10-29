import torch


class LatentRecorder:
    """Records the UNet mid-block (bottleneck) output at every denoising step.

    Attaches a forward hook to ``unet.mid_block``, so an unmodified ``diffusers``
    install is enough. Under classifier-free guidance the UNet sees a batch
    ``[uncond, text]``; only the text-conditioned half is kept.

    Usage::

        with LatentRecorder(pipe.unet, classifier_free_guidance=True) as rec:
            pipe(prompt)
        latents = rec.stack()  # [T, B, 1280, 8, 8]
    """

    def __init__(self, unet, classifier_free_guidance=True):
        self.unet = unet
        self.classifier_free_guidance = classifier_free_guidance
        self.latents = []
        self._handle = None

    def _hook(self, module, inputs, output):
        if self.classifier_free_guidance:
            output = output.chunk(2)[1]
        self.latents.append(output.detach())

    def reset(self):
        self.latents = []

    def stack(self):
        return torch.stack(self.latents)

    def __enter__(self):
        self.reset()
        self._handle = self.unet.mid_block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
