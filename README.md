---
title: "PersonaGen"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "5.44.1"
app_file: "app.py"
pinned: true
---

# PersonaGen – AI-Generated Portraits with Stable Diffusion

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face Demo](https://img.shields.io/badge/demo-Hugging%20Face-orange.svg)](https://huggingface.co/spaces/inesruizblach/PersonaGen)

---

## 📌 Overview  
**PersonaGen** is a generative AI project that creates **synthetic human portraits** using [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion) and the [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) library.  

The system takes a text prompt as input and produces high-quality faces that can be customised by **age, gender, expression, and artistic style**.  

👉 Try the **live demo on Hugging Face Spaces**: [PersonaGen Demo](https://huggingface.co/spaces/inesruizblach/PersonaGen)  

---

## 🎯 Features  
- Generate photorealistic or stylised portraits from text prompts  
- Control over **age, gender, mood, and style**  
- Artistic filters: *watercolour, comic, cyberpunk, oil painting*  
- Deployed with **Gradio** on Hugging Face Spaces  

---

## 🛠️ Tech Stack  

- **Python 3.10+** – primary language for AI development and web app.
- **PyTorch** – tensor computations, model execution on CPU/GPU. ([pytorch.org](https://pytorch.org))
- **Hugging Face Diffusers** – Stable Diffusion pipelines, model management, inference. ([github.com/huggingface/diffusers](https://github.com/huggingface/diffusers))
- **Gradio** – easy-to-use UI for running ML models in a web browser. ([gradio.app](https://gradio.app))
- **PEFT (Parameter-Efficient Fine-Tuning)** – apply LoRA weights to enhance models with minimal compute. ([huggingface.co/docs/peft](https://huggingface.co/docs/peft/index))
- **LoRA Weights for Stable Diffusion** – optional fine-tuning for realistic facial features.
- **Additional Python Libraries**:
  - `numpy` – numerical operations.
  - `Pillow` – image creation and manipulation.
  - `torchvision` – image preprocessing and utilities.
  - `ipywidgets` – interactive notebook controls (optional for demos).

---

## 🗂️ Repository Structure  

```text
PersonaGen/
├── app.py                  # Gradio interface (main entry point)
├── requirements.txt        # Dependencies for Hugging Face Space
├── README.md               # Project documentation
├── .gitignore
├── examples/               # Pre-generated demo portraits
│   ├── woman_blonde.png
│   ├── smiling_kid.png
    └── young_male.png
```

---

## ⚡ Quick Start  

Follow these steps to run PersonaGen locally or on Hugging Face Spaces:

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/PersonaGen.git
cd PersonaGen
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
python app.py
```

4. **Open the app in your browser**

* Visit the local URL shown in the console (usually http://127.0.0.1:7860).

* Or use the Share link provided by Gradio to access the app remotely.

5. **Optional: Enable LoRA for enhanced faces**

* Install PEFT if not already installed:
```bash
pip install peft
```
* Check the "Apply Face LoRA" box in the interface to use the LoRA weights.


---

## 🚀 Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/PersonaGen.git
cd PersonaGen
pip install -r requirements.txt
```

Or using conda:
```bash
conda create -n personagen python=3.10 -y
conda activate personagen
pip install -r requirements.txt
```

### Run the Gradio app locally:
```bash
python app.py
```

#### Using LoRA (optional)

Ensure the peft library is installed (pip install peft).
Enable the "Apply Face LoRA" checkbox in the app for enhanced facial features.

---

## 🎨 Usage

1. Launch the app with:
```bash
python app.py
```
2. Enter a text prompt in the Gradio interface.
Example prompts:
* `Modern portrait of woman with blonde hair, smiling, realistic`
* `Stylised portrait of a young boy smiling, beach background, watercolor style`
* `Portrait of a young male character, smiling, vibrant colours`

3. Adjust settings:
* **Steps**: number of diffusion steps (default: 25)
* **Style**: choose from photorealistic, watercolor, comic, cyberpunk, oil painting
* **LoRA**: check the box to enhance facial features (requires `peft`)

4. Generated images are displayed in the web UI and can be downloaded by clicking the image download icon.
---

## 🖼️ Example Portraits

| Prompt                                                         | Result with LoRA                                | Result without LoRA                                |
| -------------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| *Modern portrait of woman with blonde hair, smiling, realistic* | <img src="examples/blonde_woman.png" width="200"/> | <img src="examples/blonde_woman_no_lora.png" width="200"/> |
| *Stylised portrait of a young boy smiling, beach background, watercolor* | <img src="examples/young_kid_smiling.png" width="200"/> | <img src="examples/young_kid_smiling_no_lora.png" width="200"/> |

*Caption: Images in the "Result with LoRA" column include LoRA fine-tuning for enhanced facial details, while "Result without LoRA" shows the base Stable Diffusion output.*


Notes:
* Steps = number of diffusion steps (default: 25).
* CPU runtime varies depending on hardware (around 451.4s).
* LoRA enhancement requires peft and is optional.

This version is **compact, public-repo friendly**, and highlights LoRA, CPU optimization, and basic usage instructions.  

---

## 🧩 References (Models & Weights)

- **Stable Diffusion v1.5** – base text-to-image model  
  [https://huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)

- **LCM LoRA for SD v1.5** – optional LoRA weights for enhanced faces  
  [https://huggingface.co/latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)