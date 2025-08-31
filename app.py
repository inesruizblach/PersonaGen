import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
import torch

# Device and data type
DEVICE = "cpu"
DTYPE = torch.float32  # or torch.bfloat16 if your CPU supports

# Model config
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
LORA_REPO = "prithivMLmods/Qwen-Image-Synthetic-Face"

# Cache pipeline
PIPELINE = None

def load_pipeline(use_lora: bool = False):
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = StableDiffusionInpaintPipeline.from_pretrained(
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
def generate_face(prompt, init_image, mask_image, steps=35, guidance=7.5, lora=False):
    """
    Generate or modify a face using SD 2 inpainting model.
    - prompt: text describing the change/new face
    - init_image: base image (PIL)
    - mask_image: white = area to keep, black = area to inpaint
    - steps: inference steps
    - guidance: classifier-free guidance
    - lora: whether to apply LoRA weights
    """
    pipe = load_pipeline(use_lora=lora)

    result = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]
    return result


# Gradio UI
iface = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g., Make the person smile, add glasses"),
        gr.Image(label="Init Image", type="pil"),
        gr.Image(label="Mask Image", type="pil"),
        gr.Slider(10, 50, value=35, step=5, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Checkbox(label="Apply Face LoRA", value=False),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="CPU-Friendly Face Inpainting",
    description="Edit or generate faces with Stable Diffusion 2 Inpainting. Provide an init image + mask where you want changes.",
)

if __name__ == "__main__":
    iface.launch()
