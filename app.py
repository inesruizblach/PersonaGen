import gradio as gr
from PIL import Image
import torch
from diffusers import DiffusionPipeline, QwenImageEditPipeline
import numpy as np

# Device and data type
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # CPU-friendly

# Cache pipelines so we don't reload every call
PIPELINES = {"txt2img": None, "img2img": None}

# Model identifiers
TXT2IMG_MODEL = "Qwen/Qwen-Image"  # lightweight CPU model
IMG2IMG_MODEL = "Qwen/Qwen-Image-Edit"

# Positive prompt enhancement
POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."

# Predefined aspect ratios
ASPECT_RATIOS = {
    "1:1": (512, 512),
    "16:9": (512, 288),
    "9:16": (288, 512),
    "4:3": (512, 384),
    "3:4": (384, 512),
    "3:2": (512, 341),
    "2:3": (341, 512),
}

def ensure_pil_image(img):
    """
    Convert input to RGB PIL.Image.
    Supports PIL.Image, NumPy arrays, and JPG/PNG uploads.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    else:
        raise ValueError("Unsupported image format. Upload PNG/JPEG or PIL.Image.")

def load_pipeline(txt2img=True):
    """
    Load and cache CPU-friendly pipelines:
    - txt2img=True -> Qwen/Qwen-Image-Lightning
    - txt2img=False -> Qwen/Qwen-Image-Edit
    """
    key = "txt2img" if txt2img else "img2img"
    if PIPELINES[key] is None:
        if txt2img:
            PIPELINES[key] = DiffusionPipeline.from_pretrained(
                TXT2IMG_MODEL,
                torch_dtype=DTYPE
            ).to(DEVICE)
        else:
            PIPELINES[key] = QwenImageEditPipeline.from_pretrained(
                IMG2IMG_MODEL,
                torch_dtype=DTYPE
            ).to(DEVICE)
            PIPELINES[key].set_progress_bar_config(disable=None)
    return PIPELINES[key]

def generate_image(prompt, steps, aspect, cfg_scale, seed, init_image=None):
    """
    Generate an image from a text prompt or an initial image.
    
    Args:
        prompt (str): Text describing the image.
        steps (int): Number of inference steps.
        aspect (str): Aspect ratio key (for text-to-image only).
        cfg_scale (float): Guidance scale for generation.
        seed (int): Random seed; -1 for random.
        init_image (PIL.Image, optional): Input image for image-to-image.
    
    Returns:
        PIL.Image: Generated image.
    """
    # Set random seed for reproducibility
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed > -1 else None

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

# Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe your image..."),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Dropdown(list(ASPECT_RATIOS.keys()), value="16:9", label="Aspect Ratio (txt2img only)"),
        gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG Scale"),
        gr.Number(value=-1, label="Seed (-1 = random)"),
        gr.Image(label="Init Image (optional, triggers image-to-image)", type="pil")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="PersonaGen – Qwen CPU Image Generator",
    description="Lightweight CPU-friendly Qwen: text-to-image or image-to-image. Upload an image to trigger editing mode."
)

if __name__ == "__main__":
    iface.launch()
