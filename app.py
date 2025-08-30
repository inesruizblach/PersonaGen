import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Model config
MODEL_ID = "stabilityai/sd-turbo"  # CPU-friendly version
DEVICE = "cpu"
DTYPE = torch.float32  # float32 for CPU

# Cache pipeline
PIPELINE = None

def load_pipeline():
    """
    Load and cache the Stable Diffusion CPU pipeline.
    
    Returns:
        StableDiffusionPipeline: Loaded pipeline on CPU.
    """
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE
        ).to(DEVICE)
        # Enable attention slicing to reduce memory footprint
        PIPELINE.enable_attention_slicing()
    return PIPELINE

def generate_image(prompt, steps, guidance):
    """
    Generate an image from a text prompt using Stable Diffusion CPU pipeline.

    Args:
        prompt (str): Text description of the image.
        steps (int): Number of inference steps.
        guidance (float): CFG scale for generation.

    Returns:
        PIL.Image: Generated image.
    """
    pipe = load_pipeline()
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        height=256,
        width=256
    ).images[0]
    return image

# Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe your image..."),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="PersonaGen – CPU Stable Diffusion",
    description="Generate synthetic images using Stable Diffusion on CPU (text-to-image only)."
)

if __name__ == "__main__":
    iface.launch()
