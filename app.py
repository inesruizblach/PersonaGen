import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import os

# Detect device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face token (from Space secrets)
hf_token = os.getenv("HF_TOKEN")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=hf_token
)
pipe = pipe.to(device)

def generate_face(prompt, guidance=7.5, steps=25):
    """Generate a face image from a text prompt"""
    image = pipe(
        prompt,
        guidance_scale=guidance,
        num_inference_steps=steps
    ).images[0]
    return image

# Set up Gradio interface
face_demo = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g. Portrait of a 25-year-old woman, cinematic style"),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps")
    ],
    outputs=gr.Image(type="pil", label="Generated Portrait"),
    title="PersonaGen – AI Portrait Generator",
    description="Enter a text prompt to generate a synthetic human face."
)

# Launch the Gradio app
if __name__ == "__main__":
    face_demo.launch()
