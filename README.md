---
title: "PersonaGen"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "4.44.1"
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

3. **Install dependencies**
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

## 🖼️ Example Usage with Results & Metrics 📊

| Output File          | Prompt                                       | Steps |
| -------------------- | -------------------------------------------- | ----- |
| `woman_blonde.png`   | Portrait of woman with blonde hair           | 25    |
| `smiling_kid.png`    | Portrait of a young boy smiling              | 25    |
| `young_male.png`     | Portrait of a young brunette male           | 25    |

Notes:
* Steps = number of diffusion steps (default: 25).
* CPU runtime varies depending on hardware (around 372.8s).
* LoRA enhancement requires peft and is optional.

This version is **compact, public-repo friendly**, and highlights LoRA, CPU optimization, and basic usage instructions.  