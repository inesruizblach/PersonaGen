import gradio as gr
from diffusers import StableDiffusionPipeline
import torch


# Device and data type
DEVICE = "cpu"
DTYPE = torch.float32

# Model config
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
LORA_REPO = "prithivMLmods/Qwen-Image-Synthetic-Face"

# Cache pipeline
PIPELINE = None

def load_pipeline(use_lora: bool = False):
    """
    Load the Stable Diffusion v1.5 pipeline on CPU.
    Optionally applies a LoRA for realistic faces.
    
    Args:
        use_lora (bool): If True, apply LoRA weights for synthetic face generation.
    
    Returns:
        StableDiffusionPipeline: Ready-to-use pipeline.
    """
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE
        ).to(DEVICE)

    PIPELINE.enable_attention_slicing()
    PIPELINE.enable_vae_slicing()

    if use_lora:
        try:
            PIPELINE.load_lora_weights(LORA_REPO)
            print("✅ LoRA weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load LoRA weights: {e}")

    return PIPELINE

# Generate image function
def generate_face(prompt, steps=35, guidance=7.5, lora=False):
    """
    Generate a synthetic human face on CPU using SD Turbo.
    - prompt: text prompt describing the face
    - steps: inference steps (higher = better quality)
    - guidance: classifier-free guidance
    - lora: whether to apply LoRA weights
    """
    pipe = load_pipeline(use_lora=lora)
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=512,
        width=512
    ).images[0]
    return image

# Gradio UI
iface = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g., Portrait of a smiling young woman, cinematic"),
        gr.Slider(10, 50, value=35, step=5, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Checkbox(label="Apply Face LoRA", value=False)
    ],
    outputs=gr.Image(type="pil", label="Generated Face"),
    title="CPU-Friendly Face Generator",
    description="Generate realistic human faces using Stable Diffusion Turbo on CPU. Enable LoRA for better face quality."
)

if __name__ == "__main__":
    iface.launch()
