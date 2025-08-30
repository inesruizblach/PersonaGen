import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# -----------------------------
# Define available models
# -----------------------------
MODELS = {
    "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion Turbo": "stabilityai/sd-turbo",
    "DreamShaper (SD v1.5 finetune)": "Lykon/dreamshaper"
}

# Optional LoRA weights per model
LORA_PATHS = {
    "Stable Diffusion v1.5": "path/to/sd_v1-5_lora.safetensors",
    "DreamShaper (SD v1.5 finetune)": "path/to/dreamshaper_lora.safetensors"
}

# -----------------------------
# Load SD pipeline dynamically
# -----------------------------
def load_pipeline(model_key, lora=False):
    """
    Load a Stable Diffusion pipeline given a model key.
    Optionally apply LoRA weights if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = MODELS[model_key]

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    ).to(device)

    if lora and model_key in LORA_PATHS:
        pipe.unet.load_lora_weights(LORA_PATHS[model_key])

    return pipe

# -----------------------------
# Image-to-Image helper
# -----------------------------
def img2img(pipe, init_image, prompt, guidance, steps, negative_prompt, strength):
    """
    Run image-to-image pipeline using a given Stable Diffusion pipeline.
    """
    if not isinstance(init_image, Image.Image):
        init_image = Image.fromarray(init_image)

    img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        pipe.model_id,
        torch_dtype=torch.float32
    ).to(pipe.device)

    return img_pipe(
        prompt=prompt,
        init_image=init_image,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        negative_prompt=negative_prompt or None
    ).images[0]

# -----------------------------
# Generate face
# -----------------------------
def generate_face(model_key, prompt, guidance, steps, negative_prompt, init_image=None, strength=0.7, lora=False):
    """
    Generate a synthetic human face from text and optional image conditioning.
    """
    pipe = load_pipeline(model_key, lora=lora)

    if init_image:
        # Use Image-to-Image if init_image is provided
        image = img2img(pipe, init_image, prompt, guidance, steps, negative_prompt, strength)
    else:
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            negative_prompt=negative_prompt or None,
            height=256, width=256
        ).images[0]

    return image

# -----------------------------
# Gradio interface
# -----------------------------
iface = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Dropdown(list(MODELS.keys()), value="Stable Diffusion v1.5", label="Model"),
        gr.Textbox(label="Prompt", placeholder="e.g. Portrait of a 25-year-old woman, cinematic style"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Textbox(label="Negative Prompt", placeholder="e.g. blurry, low-quality, extra limbs"),
        gr.Image(label="Init Image (optional)", type="pil"),
        gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Strength (for image-to-image)"),
        gr.Checkbox(label="Use LoRA weights", value=False)
    ],
    outputs=gr.Image(type="pil", label="Generated Portrait"),
    title="PersonaGen – CPU-Friendly AI Portrait Generator",
    description="Generate synthetic human faces with optional image conditioning and LoRA fine-tuning."
)

if __name__ == "__main__":
    iface.launch()
