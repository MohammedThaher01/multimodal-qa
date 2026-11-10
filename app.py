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
        return "Error in Stage 1 (YOLO)", error_msg

    try:
        # Stage 2: The Brain (Transformer) -> Reason over data + image
        progress(0.6, desc="Stage 2: Gemini Multimodal Reasoning...")
        print("DEBUG: Starting Gemini analysis...")
        final_answer = qa_system.ask_question(image, yolo_metadata, question)
        print("DEBUG: Gemini analysis complete.")
        
        progress(1.0, desc="Analysis Complete!")
        
        # Add visual separation markers for the UI
        styled_yolo = f"## Stage 1: Spatial Metadata\n\n{yolo_metadata}"
        styled_answer = f"## Stage 2: AI Reasoning\n\n{final_answer}"
        
        return styled_yolo, styled_answer
    except Exception as e:
        error_msg = f"ERROR during Gemini analysis: {str(e)}"
        print(f"DEBUG: {error_msg}")
        import traceback
        traceback.print_exc()
        # Ensure we still return something valid for the UI to display
        return f"## Stage 1: Spatial Metadata\n\n{yolo_metadata}", f"### ERROR during Stage 2\n\n{error_msg}"


CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
#hero { text-align: center; padding: 30px 20px 6px; }
#hero h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
    background: linear-gradient(90deg, #4f46e5, #9333ea); -webkit-background-clip: text;
    background-clip: text; color: transparent; }
#hero p { color: #64748b; font-size: 1rem; margin: 10px 0 0; }
#pipeline { display: flex; justify-content: center; align-items: center; gap: 10px;
    flex-wrap: wrap; margin-top: 16px; }
#pipeline span.pill { font-size: .8rem; font-weight: 600; padding: 7px 14px;
    border-radius: 999px; background: #eef2ff; color: #4338ca; border: 1px solid #e0e7ff; }
#pipeline span.arrow { color: #a5b4fc; font-weight: 700; }
.card { border: 1px solid #e2e8f0 !important; border-radius: 16px !important;
    background: #ffffff !important; padding: 6px 8px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .05); }
.card-head { font-weight: 700; color: #0f172a; padding: 4px 2px 8px; font-size: .95rem; }
#analyze-btn, #analyze-btn button { font-weight: 700 !important; border-radius: 10px !important; }
.gradio-container .secondary { border-radius: 10px !important; }
#footer { text-align: center; color: #94a3b8; font-size: .8rem; padding: 20px 0 8px; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]),
    css=CSS,
    title="Grounded Multimodal Visual Q&A",
) as demo:
    gr.HTML("""
    <div id="hero">
      <h1>Grounded Multimodal Visual Q&amp;A</h1>
      <p>Deterministic object detection meets grounded multimodal reasoning.</p>
      <div id="pipeline">
        <span class="pill">YOLOv8 Medium</span>
        <span class="arrow">&#8594;</span>
        <span class="pill">Gemini 3.5 Flash</span>
      </div>
    </div>
    """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=2, elem_classes="card"):
            input_image = gr.Image(type="pil", label="Upload Image", interactive=True, height=300)
            input_question = gr.Textbox(
                label="2. Ask a question", 
                placeholder="e.g., How many objects are in the top-left?",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze Scene", variant="primary", elem_id="analyze-btn", scale=4)
                clear_btn = gr.ClearButton([input_image, input_question], value="Clear", scale=1)
            gr.Examples(
                label="Try an example",
                examples=[
                    ["test_images/pexels-bruno-brandao-727244277-18880069.jpg", "How many people are in this scene?"],
                    ["test_images/noname_13-traffic-3612474_1920.jpg", "How many vehicles are in the top-left?"],
                ],
                inputs=[input_image, input_question],
            )
            
        # Right Column: Outputs
        with gr.Column(scale=3):
            with gr.Group():
                gr.HTML("<div class='card-head'>Stage 1 &middot; Spatial Metadata</div>")
                output_metadata = gr.Markdown(value="*Results will appear here after analysis*")
            
            gr.HTML("<div style='margin: 20px 0; border-bottom: 1px solid #ddd;'></div>") # Better visual divider
            
            with gr.Group():
                gr.HTML("<div class='card-head'>Stage 2 &middot; AI Reasoning</div>")
                output_answer = gr.Markdown(value="*AI reasoning will appear here after analysis*")


            
    # Connect the button to our pipeline function
    submit_btn.click(
        fn=process_qa,
        inputs=[input_image, input_question],
        outputs=[output_metadata, output_answer],
        show_progress="full"
    )

    gr.HTML("<div id='footer'>YOLOv8 Medium &middot; Gemini 3.5 Flash &middot; Built with Gradio &amp; LangChain</div>")

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
