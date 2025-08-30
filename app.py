import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import numpy as np


# Model config
MODEL_ID = "stabilityai/sd-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache dict so we don't reload every call
PIPELINES = {}


def ensure_pil_image(img):
    """
    Ensure the input is converted to a valid RGB PIL.Image.
    Supports PIL.Image, NumPy arrays, and common formats like JPG/PNG.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    else:
        raise ValueError("⚠️ Unsupported image format. Please upload PNG or JPEG.")


def load_pipeline(img2img=False, lora_repo=None):
    """
    Load and cache the correct pipeline.
    If img2img=True, load StableDiffusionImg2ImgPipeline.
    Else, load StableDiffusionPipeline (text-to-image).
    Optionally apply LoRA weights from Hugging Face repo.
    """
    key = "img2img" if img2img else "txt2img"

    if key not in PIPELINES:
        if img2img:
            PIPELINES[key] = StableDiffusionImg2ImgPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32
            ).to(device)
        else:
            PIPELINES[key] = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32
            ).to(device)

    pipe = PIPELINES[key]

    # Apply LoRA if a repo is given
    if lora_repo:
        try:
            pipe.unet.load_lora_weights(lora_repo)
        except Exception as e:
            print(f"⚠️ Failed to load LoRA weights from {lora_repo}: {e}")

    return pipe


def generate_image(prompt, guidance, steps, init_image=None, strength=0.7, use_lora=False, lora_repo=None):
    """
    Generate an image from a prompt (text-to-image).
    If init_image is provided, run image-to-image instead.
    Optionally apply LoRA weights.
    """
    if init_image:
        init_image = ensure_pil_image(init_image)
        pipe = load_pipeline(img2img=True, lora_repo=lora_repo if use_lora else None)
        result = pipe(
            prompt=prompt,
            init_image=init_image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
        ).images[0]
    else:
        pipe = load_pipeline(img2img=False, lora_repo=lora_repo if use_lora else None)
        result = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=256, width=256
        ).images[0]

    return result


# Gradio UI
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g. Portrait of a 25-year-old woman, cinematic style"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Image(label="Init Image (optional)", type="pil"),
        gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Strength (for image-to-image)"),
        gr.Checkbox(label="Use LoRA weights", value=False),
        gr.Textbox(label="LoRA HF Repo (optional)", placeholder="e.g. someuser/sd-lora-face")
    ],
    outputs=gr.Image(type="pil", label="Generated Portrait"),
    title="PersonaGen – AI Portrait Generator",
    description="Generate synthetic human faces with Stable Diffusion Turbo. Upload an image for image-to-image mode (supports PNG/JPEG). Optionally apply LoRA weights from Hugging Face."
)

if __name__ == "__main__":
    iface.launch()
