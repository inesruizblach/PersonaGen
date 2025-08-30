import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

MODEL_ID = "stabilityai/sd-turbo"
DEVICE = "cpu"

PIPELINE = None

def load_pipeline():
    """
    Load and cache Stable Diffusion pipeline for CPU.
    Returns:
        StableDiffusionPipeline: ready-to-use pipeline.
    """
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        ).to(DEVICE)
        PIPELINE.enable_attention_slicing()
    return PIPELINE

def generate_image(prompt, steps, guidance):
    """
    Generate an image from a text prompt using CPU.
    Args:
        prompt (str): Description of the image.
        steps (int): Inference steps.
        guidance (float): CFG scale.
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

iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe your image..."),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="PersonaGen – CPU Stable Diffusion",
    description="CPU-friendly text-to-image generation using Stable Diffusion Turbo."
)

if __name__ == "__main__":
    iface.launch()
