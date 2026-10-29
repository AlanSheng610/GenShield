import torch

from .hooks import LatentRecorder


def shielded_generate(pipe, shield, prompt, seed=0, guidance_scale=7.5, num_inference_steps=50):
    """Generate with a ``StableDiffusionPipeline`` and stop as soon as ``shield`` flags a concept.

    Returns:
        (image or None, report). The image is None if generation was blocked; the report
        holds the step, concept and score that triggered the block, and the maximum
        similarity reached for every concept.
    """
    report = {"blocked": False, "step": None, "concept": None, "score": None, "max_scores": None}

    with LatentRecorder(pipe.unet, classifier_free_guidance=guidance_scale > 1) as rec:
        def on_step_end(pipeline, step, timestep, callback_kwargs):
            scores = shield.score(rec.latents[-1], step)[0]  # [K]
            report["max_scores"] = scores if report["max_scores"] is None else torch.maximum(report["max_scores"], scores)
            k = int(scores.argmax())
            if scores[k] >= shield.threshold:
                report.update(blocked=True, step=step, concept=shield.concepts[k], score=float(scores[k]))
                pipeline._interrupt = True  # remaining denoising steps are skipped
            rec.reset()
            return callback_kwargs

        out = pipe(prompt, generator=torch.Generator("cpu").manual_seed(seed),
                   guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                   callback_on_step_end=on_step_end,
                   output_type="latent")

    report["max_scores"] = dict(zip(shield.concepts, report["max_scores"].tolist()))
    if report["blocked"]:
        return None, report
    with torch.no_grad():
        image = pipe.vae.decode(out.images / pipe.vae.config.scaling_factor).sample
    return pipe.image_processor.postprocess(image)[0], report
