import gradio as gr
from PIL import Image
from extractor import SpatialExtractor
from llm_chain import VisionQASystem
import traceback

# 1. Global Initialization (The "Lazy Load" Trick)
# We set these to None so the server boots instantly in 0.1 seconds.
# This prevents the Hugging Face UI from timing out and getting stuck on "Starting".
extractor = None
qa_system = None

def process_qa(image, question, progress=gr.Progress()):
    global extractor, qa_system
    
    # Lazy load the heavy AI models ONLY when the user clicks the button for the first time
    if extractor is None or qa_system is None:
        progress(0.1, desc="Booting AI Engines (First run takes ~30 seconds)...")
        print("DEBUG: Lazy loading AI engines...")
        extractor = SpatialExtractor()
        qa_system = VisionQASystem()
        print("DEBUG: Engines Ready!")

    print(f"DEBUG: Processing started. Question: {question}")
    if image is None:
        print("DEBUG: No image provided.")
        return "Please upload an image.", "No image provided."
    if not question.strip():
        print("DEBUG: No question provided.")
        return "Please ask a question.", "No question provided."
        
    try:
        # Stage 1: The Eyes (CNN) -> Get absolute spatial data
        progress(0.4, desc="Stage 1: YOLOv8 Spatial Analysis...")
        print("DEBUG: Starting YOLO analysis...")
        yolo_metadata = extractor.analyze_image(image)
        print(f"DEBUG: YOLO metadata extracted: {yolo_metadata[:50]}...")
    except Exception as e:
        error_msg = f"ERROR during YOLO analysis: {str(e)}"
        print(f"DEBUG: {error_msg}")
        traceback.print_exc()
        return "⚠️ Error in Stage 1 (YOLO)", error_msg

    try:
        # Stage 2: The Brain (Transformer) -> Reason over data + image
        progress(0.8, desc="Stage 2: Gemini Multimodal Reasoning...")
        print("DEBUG: Starting Gemini analysis...")
        final_answer = qa_system.ask_question(image, yolo_metadata, question)
        print("DEBUG: Gemini analysis complete.")
        
        progress(1.0, desc="Analysis Complete!")
        
        # Add visual separation markers for the UI
        styled_yolo = f"## 🔍 Stage 1: Spatial Metadata\n\n{yolo_metadata}"
        styled_answer = f"## 🧠 Stage 2: AI Reasoning\n\n{final_answer}"
        
        return styled_yolo, styled_answer
    except Exception as e:
        error_msg = f"ERROR during Gemini analysis: {str(e)}"
        print(f"DEBUG: {error_msg}")
        traceback.print_exc()
        # Fallback to ensure we still return something valid for the UI to display
        meta = yolo_metadata if 'yolo_metadata' in locals() else "Metadata extraction failed."
        return f"## 🔍 Stage 1: Spatial Metadata\n\n{meta}", f"### ⚠️ ERROR during Stage 2\n\n{error_msg}"


# 2. Build a professional, two-column UI using Gradio Blocks
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])) as demo:
    gr.Markdown("# 🧠 Grounded Multimodal Visual Q&A")
    gr.Markdown("Architecture: **YOLOv8 Medium** (Spatial Verification) ➔ **Gemini 3.5 Flash** (Cognitive Reasoning)")
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=2):
            input_image = gr.Image(type="pil", label="1. Upload Image", interactive=True)
            input_question = gr.Textbox(
                label="2. Ask a question", 
                placeholder="e.g., How many objects are in the top-left?",
                lines=2
            )
            submit_btn = gr.Button("🚀 Analyze Scene", variant="primary")
            
        # Right Column: Outputs
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### 🔍 Stage 1: Spatial Metadata")
                output_metadata = gr.Markdown(value="*Results will appear here after analysis*")
            
            gr.HTML("<div style='margin: 20px 0; border-bottom: 1px solid #ddd;'></div>") # Better visual divider
            
            with gr.Group():
                gr.Markdown("### 🧠 Stage 2: AI Reasoning")
                output_answer = gr.Markdown(value="*AI reasoning will appear here after analysis*")
            
    # Connect the button to our pipeline function
    submit_btn.click(
        fn=process_qa,
        inputs=[input_image, input_question],
        outputs=[output_metadata, output_answer]
    )


# 3. Launch the Server
if __name__ == "__main__":
    demo.launch(ssr_mode=False)