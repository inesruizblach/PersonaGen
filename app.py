import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# --- Load text-to-image pipeline (always needed) ---
def load_text2img():
    """Initialize the base text-to-image Stable Diffusion pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    )
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()  # reduce memory usage
    return pipe.to(device)

text_pipe = load_text2img()


# --- Image-to-image function (loaded only when needed) ---
def run_img2img(prompt, negative_prompt, init_image, strength, guidance, steps, device):
    """
    Run image-to-image generation with Stable Diffusion.
    Loads the Img2Img pipeline only when required.
    """
    # Ensure init_image is a PIL Image
    if not isinstance(init_image, Image.Image):
        init_image = Image.fromarray(init_image)

    # Lazy-load img2img pipeline
    img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
    ).to(device)
    img_pipe.enable_attention_slicing()

    # Generate variation from init_image
    result = img_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=init_image,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images[0]

    # Free pipeline memory
    del img_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None  

    return result


# --- Unified generation function ---
def generate_face(prompt, guidance=7.5, steps=25, negative_prompt="", init_image=None, strength=0.7):
    """
    Generate an AI portrait.
    - If `init_image` is provided → image-to-image mode.
    - Otherwise → pure text-to-image.
    """
    device = text_pipe.device
    if init_image is not None:
        return run_img2img(prompt, negative_prompt, init_image, strength, guidance, steps, device)
    else:
        return text_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=256, width=256,  # lower res = faster on CPU
        ).images[0]


# --- Gradio interface ---
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
