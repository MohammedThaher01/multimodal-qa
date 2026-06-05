---
title: Multimodal Visual QA
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.16.0
python_version: '3.10'
app_file: app.py
pinned: false
---

# Building a Smart Visual Counting & Reasoning System

This project demonstrates a "Grounded AI" approach to visual question answering. By combining the deterministic object detection capabilities of YOLOv8 with the reasoning power of Gemini 3.5 Flash, we solve the common problem of LLM hallucinations when counting or locating objects in an image.

## How it Works
1. **The Eyes (YOLOv8m):** Analyzes the image to find objects and their exact coordinates.
2. **The Brain (Gemini 3.5 Flash):** Uses the image and the raw data from YOLO to provide a factually grounded answer.

## Key Features
- **Deterministic Counting:** Never miscounts objects thanks to YOLOv8 integration.
- **Spatial Awareness:** Understands exactly where objects are located (Top-Left, Bottom-Right, etc.).
- **Gradio Interface:** Professional UI with real-time progress indicators and Markdown support.

## Installation & Setup
1. Clone the repository.
2. Create a `.env` file with your `GOOGLE_API_KEY`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the app: `python app.py`.