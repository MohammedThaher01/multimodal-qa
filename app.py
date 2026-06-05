import gradio as gr
from PIL import Image
from extractor import SpatialExtractor
from llm_chain import VisionQASystem

# 1. Initialize our two engines globally so they don't reload on every click
print("Initializing AI Engines... This might take a few seconds.")
extractor = SpatialExtractor()
qa_system = VisionQASystem()
print("Engines Ready!")

def process_qa(image, question, progress=gr.Progress()):
    print(f"DEBUG: Processing started. Question: {question}")
    if image is None:
        print("DEBUG: No image provided.")
        return "Please upload an image.", "No image provided."
    if not question.strip():
        print("DEBUG: No question provided.")
        return "Please ask a question.", "No question provided."
        
    try:
        # Stage 1: The Eyes (CNN) -> Get absolute spatial data
        progress(0.2, desc="Stage 1: YOLOv8 Spatial Analysis...")
        print("DEBUG: Starting YOLO analysis...")
        yolo_metadata = extractor.analyze_image(image)
        print(f"DEBUG: YOLO metadata extracted: {yolo_metadata[:50]}...")
    except Exception as e:
        error_msg = f"ERROR during YOLO analysis: {str(e)}"
        print(f"DEBUG: {error_msg}")
        return "⚠️ Error in Stage 1 (YOLO)", error_msg

    try:
        # Stage 2: The Brain (Transformer) -> Reason over data + image
        progress(0.6, desc="Stage 2: Gemini Multimodal Reasoning...")
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
        import traceback
        traceback.print_exc()
        # Ensure we still return something valid for the UI to display
        return f"## 🔍 Stage 1: Spatial Metadata\n\n{yolo_metadata}", f"### ⚠️ ERROR during Stage 2\n\n{error_msg}"


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

if __name__ == "__main__":
    # Standard launch for Hugging Face Spaces
    demo.launch()
