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
│   ├── adult_female.png
│   ├── young_child.png
│   ├── senior_male.png
│   ├── domestic_cat.png
│   └── dragon_fantasy.png
├── notebooks/              # Notebook demo for generating examples
    └── demo_portraits.ipynb
```

---

## 🚀 Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/PersonaGen.git
cd PersonaGen
pip install -r requirements.txt
```

Or if running with conda env:
```bash
conda create -n personagen python=3.10 -y
conda activate personagen
pip install -r requirements.txt
```

### Run the Gradio app locally:
```bash
python app.py
```

### Example: demo notebook to generate sample images

1. Open Jupyter Notebook:

```bash
jupyter notebook notebooks/demo_portraits.ipynb
```
2. Run all cells.
3. Example images will be saved automatically to examples/

---

## 🖼️ Example Usage with Results & Metrics 📊

| Output File         | Prompt                                       | Steps | Runtime 
| ------------------- | -------------------------------------------- | ----- | ------- 
| `woman_blonde.png`  | Portrait of woman with blonde hair           | 25    | 
| `smiling_kid.png`   | Portrait of a young boy smiling              | 25    | 
| `small_dog.png`     | Portrait of a yorkshire terrier              | 25    | 
| `dragon_fantasy.png`| Portrait of a fantasy dragon, vibrant colors | 25    |

**Notes:**
- `Steps` = number of inference steps (default: 25).  
- `Runtime` = measured wall-clock time per image on CPU (approx).  
