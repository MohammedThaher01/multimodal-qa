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
    
    # Enhanced Header Section
    gr.HTML("""
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px 0;">
            <h1 style="font-weight: 800; font-size: 2.2rem; margin-bottom: 8px;">🧠 Grounded Multimodal Visual Q&A</h1>
            <p style="font-size: 1.1rem; color: #4b5563;">Architecture: <b>YOLOv8 Medium</b> (Spatial Verification) ➔ <b>Gemini 3.5 Flash</b> (Cognitive Reasoning)</p>
        </div>
    """)
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=5, min_width=350):
            gr.Markdown("### 📥 Input Panel")
            input_image = gr.Image(type="pil", label="1. Upload Image", interactive=True)
            input_question = gr.Textbox(
                label="2. Ask a question", 
                placeholder="e.g., How many objects are in the top-left?",
                lines=3
            )
            submit_btn = gr.Button("🚀 Analyze Scene", variant="primary", size="lg")
            
        # Right Column: Outputs
        with gr.Column(scale=7, min_width=450):
            gr.Markdown("### 📤 Analysis Results")
            
            # Grouping gives a clean card-like appearance.
            with gr.Group():
                output_metadata = gr.Markdown(value="*Stage 1 spatial data will appear here after analysis...*")
            
            gr.HTML("<div style='margin: 16px 0;'></div>") # Subtle spacing between output cards
            
            with gr.Group():
                output_answer = gr.Markdown(value="*Stage 2 AI reasoning will appear here after analysis...*")

    # Connect the button to our pipeline function
    submit_btn.click(
        fn=process_qa,
        inputs=[input_image, input_question],
        outputs=[output_metadata, output_answer]
    )
    
    # Custom HTML Footer (No dataset link, SVG standard styling compatible with Hugging Face)
    gr.HTML("""
        <div style="text-align: center; margin-top: 60px; padding-top: 24px; border-top: 1px solid #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <p style="color: #9ca3af; font-size: 14px; font-weight: 500; letter-spacing: 0.05em; margin-bottom: 16px;">
                Developed by Mohammed Thaher S
            </p>
            <div style="display: flex; justify-content: center; gap: 24px; align-items: center; padding-bottom: 20px;">
                <a href="https://github.com/MohammedThaher01" target="_blank" rel="noreferrer" style="color: #6b7280; text-decoration: none; transition: color 0.2s;" onmouseover="this.style.color='#111827'" onmouseout="this.style.color='#6b7280'" aria-label="GitHub">
                    <svg style="width: 24px; height: 24px;" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd" />
                    </svg>
                </a>
                <a href="https://www.linkedin.com/in/mohammed-thaher-s/" target="_blank" rel="noreferrer" style="color: #6b7280; text-decoration: none; transition: color 0.2s;" onmouseover="this.style.color='#0a66c2'" onmouseout="this.style.color='#6b7280'" aria-label="LinkedIn">
                    <svg style="width: 24px; height: 24px;" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path fill-rule="evenodd" d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" clip-rule="evenodd" />
                    </svg>
                </a>
            </div>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)