import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA repository (optional fine-tuning)
LORA_REPO = "prithivMLmods/Qwen-Image-Synthetic-Face"

# Load public Stable Diffusion v1.5 model
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# CPU optimizations
if device == "cpu":
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

# Function to optionally load LoRA weights
def apply_lora(pipe, use_lora: bool):
    if use_lora:
        try:
            pipe.load_lora_weights(LORA_REPO)
            print("✅ LoRA weights applied successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load LoRA: {e}")
    return pipe

# Image generation function
def generate_face(prompt, guidance=7.5, steps=25, use_lora=False):
    """Generate a synthetic face with optional LoRA enhancement"""
    pipe_with_lora = apply_lora(pipe, use_lora)
    with torch.inference_mode():
        image = pipe_with_lora(
            prompt,
            guidance_scale=guidance,
            num_inference_steps=steps
        ).images[0]
    return image

# Gradio UI
face_demo = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g., Portrait of a 25-year-old woman, cinematic style"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Checkbox(label="Apply Face LoRA", value=False)
    ],
    outputs=gr.Image(type="pil", label="Generated Portrait"),
    title="PersonaGen – AI Portrait Generator with LoRA",
    description="Generate synthetic human faces. Enable LoRA for enhanced face quality."
)

if __name__ == "__main__":
    face_demo.launch()
