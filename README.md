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
├── app.py # Gradio interface (main entry point)
├── requirements.txt # Dependencies for Hugging Face Space
├── README.md # Project documentation
├── .gitignore
├── examples/
│ ├── sample1.png
│ ├── sample2.png
│ └── ...
├── notebooks/
│ └── training.ipynb
└── utils/
└── preprocessing.py
```

---

## 🚀 Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/PersonaGen.git
cd PersonaGen
pip install -r requirements.txt
```
Run the Gradio app locally:
```bash
python app.py
```

---

## 🖼️ Example Usage
Prompt:
Hyper-realistic portrait of a smiling 30-year-old woman, studio lighting, professional photography

Output:

## 📊 Results & Metrics
