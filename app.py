import gradio as gr
import torch
from diffusers import DiffusionPipeline

# CPU device & dtype
DEVICE = "cpu"
DTYPE = torch.float32  # float32 for CPU

# Model and LoRA
BASE_MODEL = "Qwen/Qwen-Image"
LORA_REPO = "prithivMLmods/Qwen-Image-Synthetic-Face"
TRIGGER_WORD = "Synthetic Face"

# Cache pipeline
PIPELINE = None

# Positive prompt enhancement
POSITIVE_MAGIC = ", Ultra HD, 4K, cinematic composition."

# Fixed 16:9 aspect ratio
WIDTH, HEIGHT = 512, 288

def load_pipeline(apply_lora=False):
    """
    Load and cache the Qwen CPU text-to-image pipeline.
    Optionally apply LoRA weights for synthetic face enhancement.

    Args:
        apply_lora (bool): If True, load LoRA weights from Hugging Face repo.

    Returns:
        DiffusionPipeline: Loaded pipeline.
    """
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = DiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=DTYPE
        ).to(DEVICE)
        if apply_lora:
            try:
                PIPELINE.load_lora_weights(LORA_REPO)
            except Exception as e:
                print(f"⚠️ Failed to load LoRA weights: {e}")
    return PIPELINE

def generate_image(prompt, steps, cfg_scale, use_lora):
    """
    Generate an image from a text prompt using the Qwen CPU model.

    Args:
        prompt (str): Text description of the image.
        steps (int): Number of inference steps.
        cfg_scale (float): Guidance scale.
        use_lora (bool): Whether to apply synthetic face LoRA weights.

    Returns:
        PIL.Image: Generated image.
    """
    pipe = load_pipeline(apply_lora=use_lora)
    final_prompt = f"{TRIGGER_WORD}, {prompt}" if use_lora else prompt
    image = pipe(
        prompt=final_prompt + POSITIVE_MAGIC,
        negative_prompt="",
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=steps,
        true_cfg_scale=cfg_scale
    ).images[0]
    return image

# Gradio UI
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe your image..."),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps"),
        gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG Scale"),
        gr.Checkbox(label="Apply Synthetic Face LoRA", value=False)
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="PersonaGen – Qwen CPU Text-to-Image with LoRA",
    description="Generate synthetic human faces using Qwen CPU model. Optionally apply LoRA weights for enhanced face generation."
)

if __name__ == "__main__":
    iface.launch()
