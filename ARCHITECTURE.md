
---

# **Architecture & Evaluation Report: Multimodal Visual Q&A System**

## **1. Executive Summary**

This project implements a production-grade, two-stage AI pipeline that combines high-speed computer vision (CNN) with advanced multimodal reasoning (Transformer). The system is designed to solve a critical limitation in current Large Vision-Language Models (VLMs): the tendency to hallucinate spatial metrics and miscount objects due to their reliance on semantic patch-processing rather than exact geometric mapping.

By anchoring a GenAI reasoning engine with deterministic spatial metadata, the system provides mathematically grounded, highly reliable visual analysis.

## **2. System Architecture (The "Eye & Brain" Pipeline)**

The application is decoupled into two independent stages, reflecting enterprise-level backend practices:

### **Stage 1: The Spatial Extractor (YOLOv8 Medium)**

* **Engine:** `ultralytics` YOLOv8m (`yolov8m.pt`)
* **Function:** Acts as the "Eyes." It scans the raw image at a high resolution (`imgsz=1024`, `conf=0.25`) to extract absolute geometric ground truth.
* **Output:** Translates bounding box coordinates into structured, human-readable spatial metadata (e.g., `11 person(s) detected`, `person (Bottom-Left)`).
* **Advantage:** Bypasses LLM spatial blindness by forcing deterministic counting before reasoning begins.

### **Stage 2: The Reasoning Orchestrator (LangChain + Gemini 3.5 Flash)**

* **Engine:** `langchain-google-genai` using the cutting-edge `gemini-3.5-flash` model.
* **Function:** Acts as the "Brain." It receives the raw image, the user's natural language query, and the YOLO metadata string.
* **Prompt Anchoring:** The system prompt explicitly enforces strict compliance with the CNN's data:
* *Rule 1: NEVER contradict the YOLOv8 object counts. Treat them as absolute facts.*
* *Rule 2: Use the YOLOv8 spatial layout data to understand where things are located before answering.*



### **User Interface & Backend**

* **Frontend:** Built using Gradio 6.x for a clean, two-column layout showing both the raw spatial data and the final LLM reasoning side-by-side.
* **Deployment Readiness:** Configured to run cleanly on custom ports (`7865` at `0.0.0.0`) with robust exception handling and terminal debugging for concurrent request stability.

---

## **3. System Evaluation & Testing**

### **The Good (Strengths & Successes)**

* **Zero-Hallucination Counting:** The prompt anchoring works flawlessly. In stress tests, if YOLO mathematically detects a UI element as a "sports ball," the LLM is forced to accept the classification but uses its semantic reasoning to correctly explain the discrepancy to the user (e.g., "A small white circular UI marker (sports ball) is visible").
* **Complex Contextual Reconciliation:** During the "Projector Test" (4 physical people sitting in a living room watching 10 people on a projected football game), YOLO blindly returned `14 person(s)`. Instead of failing, the LLM brilliantly synthesized the data: it confirmed the count of 14, but accurately separated the 4 physical viewers from the 10 digital players in its explanation.

### **The Drawbacks (Limitations Discovered)**

* **The "Domain Shift" Problem:** While testing against an industrial factory image, YOLOv8 failed to detect workers deep in the background. Because the pre-trained COCO dataset expects crisp photography, the blurry/compressed background pixels caused the model to drop its confidence below the threshold, resulting in undercounting (detecting 11 workers instead of 21).
* **Contextual Semantic Gaps:** Because YOLO was only extracting the `person` class, it passed no data regarding safety gear. Relying on its training bias that "factory workers wear hard hats," the LLM slightly hallucinated the presence of PPE on workers who were actually wearing baseball caps.

### **4. Future Roadmap & Scalability**

To evolve this project from a robust prototype into a deployable enterprise solution, the following architectural upgrades are mapped out:

1. **Custom Fine-Tuning:** Retrain the YOLOv8 model on a domain-specific dataset (e.g., custom classes for `helmet`, `no_helmet`, `safety_vest`) to eliminate the LLM's semantic gaps regarding safety compliance.
2. **Dynamic Thresholding:** Implement dynamic confidence and IoU (Intersection over Union) thresholds in the API request to allow users to toggle sensitivity for dense, highly occluded environments.
