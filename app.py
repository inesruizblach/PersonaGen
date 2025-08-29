import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import os

# Load Stable Diffusion model pipeline
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,          
    )
    return pipe.to(device)
pipe = load_pipeline()

def generate_face(prompt, guidance=7.5, steps=25):
    """Generate a synthetic human face based on the text prompt."""
    image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]
    return image

# Gradio interface
face_demo = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Textbox(
            label="Prompt", 
            placeholder="e.g. Portrait of a 25-year-old woman, cinematic style"
        ),
        gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(10, 50, value=25, step=5, label="Inference Steps")
    ],
    outputs=gr.Image(type="pil", label="Generated Portrait"),
    title="PersonaGen – AI Portrait Generator",
    description="Enter a text prompt to generate a synthetic human face."
)

if __name__ == "__main__":
    face_demo.launch()
