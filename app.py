import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# Load Stable Diffusion pipeline (CPU-friendly)
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )
    pipe.enable_attention_slicing()  # Save memory
    return pipe.to(device)

pipe = load_pipeline()

# To-Do: LoRA weights
# pipe.unet.load_lora_weights("path/to/lora_weights")

def generate_face(prompt, guidance=7.5, steps=25, negative_prompt="", init_image=None, strength=0.7):
    """Generate a synthetic human face with optional image-to-image conditioning."""
    if init_image:
        # Convert to PIL if not already
        if not isinstance(init_image, Image.Image):
            init_image = Image.fromarray(init_image)
        img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
        ).to(pipe.device)
        img_pipe.enable_attention_slicing()
        image = img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            init_image=init_image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=256, width=256
        ).images[0]
    return image

# Gradio interface
iface = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g. Portrait of a 25-year-old woman, cinematic style"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Textbox(label="Negative Prompt", placeholder="e.g. blurry, low-quality, extra limbs"),
        gr.Image(label="Init Image (optional)", type="pil"),
        gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Strength (for image-to-image)")
    ],
    outputs=gr.Image(type="pil", label="Generated Portrait"),
    title="PersonaGen – CPU-Friendly AI Portrait Generator",
    description="Generate synthetic human faces with optional image conditioning."
)

if __name__ == "__main__":
    iface.launch()
