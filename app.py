import gradio as gr
from PIL import Image
import torch
from diffusers import DiffusionPipeline, QwenImageEditPipeline
import numpy as np

# Device & dtype selection
device = "cuda" if torch.cuda.is_available() else "cpu"
txt2img_dtype = torch.bfloat16 if device == "cuda" else torch.float32
img2img_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Cache pipelines to avoid reloading
PIPELINES = {
    "txt2img": None,
    "img2img": None
}

# Constants
TXT2IMG_MODEL = "Qwen/Qwen-Image"
IMG2IMG_MODEL = "Qwen/Qwen-Image-Edit"
POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

# Helper Functions
def ensure_pil_image(img):
    """
    Ensure that the input is converted to a valid RGB PIL.Image.
    Supports PIL.Image, NumPy arrays, and common image formats like JPG/PNG.

    Args:
        img (PIL.Image.Image | np.ndarray): Input image.

    Returns:
        PIL.Image.Image: Converted RGB image.

    Raises:
        ValueError: If input is not a supported format.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, (np.ndarray,)):
        return Image.fromarray(img).convert("RGB")
    else:
        raise ValueError("Unsupported image format. Upload PNG/JPEG or PIL.Image.")

def load_pipeline(txt2img=True):
    """
    Load and cache the correct pipeline.

    Args:
        txt2img (bool): True for text-to-image, False for image-to-image.

    Returns:
        diffusers.Pipeline: Loaded Stable Diffusion or QwenImageEdit pipeline.
    """
    key = "txt2img" if txt2img else "img2img"
    if PIPELINES[key] is None:
        if txt2img:
            PIPELINES[key] = DiffusionPipeline.from_pretrained(
                TXT2IMG_MODEL,
                torch_dtype=txt2img_dtype
            ).to(device)
        else:
            PIPELINES[key] = QwenImageEditPipeline.from_pretrained(
                IMG2IMG_MODEL,
                torch_dtype=img2img_dtype
            ).to(device)
            PIPELINES[key].set_progress_bar_config(disable=None)
    return PIPELINES[key]

def generate_image(prompt, steps, aspect, cfg_scale, seed, init_image=None):
    """
    Generate an image from a text prompt or using an input image.

    Args:
        prompt (str): The text prompt describing the image.
        steps (int): Number of inference steps.
        aspect (str): Aspect ratio key for text-to-image.
        cfg_scale (float): CFG scale for classifier-free guidance.
        seed (int): Random seed (-1 = random).
        init_image (PIL.Image.Image | np.ndarray | None): Optional image for image-to-image.

    Returns:
        PIL.Image.Image: Generated image.
    """
    generator = torch.Generator(device=device).manual_seed(seed) if seed > -1 else None

    if init_image:
        # Image-to-image mode
        init_image = ensure_pil_image(init_image)
        pipe = load_pipeline(txt2img=False)
        output = pipe(
            image=init_image,
            prompt=prompt,
            negative_prompt="",
            true_cfg_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator
        )
        image = output.images[0]
    else:
        # Text-to-image mode
        pipe = load_pipeline(txt2img=True)
        width, height = ASPECT_RATIOS[aspect]
        image = pipe(
            prompt=prompt + POSITIVE_MAGIC,
            negative_prompt="",
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg_scale,
            generator=generator
        ).images[0]

    return image

# Gradio Interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe your image..."),
        gr.Slider(10, 100, value=50, step=5, label="Inference Steps"),
        gr.Dropdown(list(ASPECT_RATIOS.keys()), value="16:9", label="Aspect Ratio (txt2img only)"),
        gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG Scale"),
        gr.Number(value=-1, label="Seed (-1 = random)"),
        gr.Image(label="Init Image (optional, triggers image-to-image)", type="pil")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="PersonaGen – Qwen AI Image Generator",
    description="Text-to-image with Qwen/Qwen-Image, or image-to-image with Qwen/Qwen-Image-Edit. Upload an image for image editing mode."
)

if __name__ == "__main__":
    iface.launch()
