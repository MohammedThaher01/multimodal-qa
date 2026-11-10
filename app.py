
import gradio as gr
from PIL import Image
from extractor import SpatialExtractor
from llm_chain import VisionQASystem
import traceback


# ============================================================
# 1. INITIALIZE AI ENGINES
# ============================================================

print("Initializing AI Engines... This might take a few seconds.")

extractor = SpatialExtractor()
qa_system = VisionQASystem()

print("Engines Ready!")


# ============================================================
# 2. PROCESSING PIPELINE
# ============================================================

def process_qa(image, question, progress=gr.Progress()):

    print(f"DEBUG: Processing started. Question: {question}")

    if image is None:
        print("DEBUG: No image provided.")
        return "Please upload an image.", "No image provided."

    if not question or not question.strip():
        print("DEBUG: No question provided.")
        return "Please ask a question.", "No question provided."

    try:
        # Stage 1: Spatial Analysis (YOLOv8)
        progress(0.2, desc="Stage 1: YOLOv8 Spatial Analysis...")

        print("DEBUG: Starting YOLO analysis...")

        yolo_metadata = extractor.analyze_image(image)

        print(f"DEBUG: YOLO metadata extracted: {str(yolo_metadata)[:50]}...")

    except Exception as e:

        error_msg = f"ERROR during YOLO analysis: {str(e)}"

        print(f"DEBUG: {error_msg}")

        return "Error in Stage 1 (YOLO)", error_msg

    try:
        # Stage 2: Multimodal Reasoning (Gemini)
        progress(0.6, desc="Stage 2: Gemini Multimodal Reasoning...")

        print("DEBUG: Starting Gemini analysis...")

        final_answer = qa_system.ask_question(
            image,
            yolo_metadata,
            question
        )

        print("DEBUG: Gemini analysis complete.")

        progress(1.0, desc="Analysis Complete!")

        # Format outputs
        styled_yolo = f"{yolo_metadata}"
        styled_answer = f"{final_answer}"

        return styled_yolo, styled_answer

    except Exception as e:

        error_msg = f"ERROR during Gemini analysis: {str(e)}"

        print(f"DEBUG: {error_msg}")

        traceback.print_exc()

        return (
            f"{yolo_metadata}",
            f"### Error during Stage 2\n\n{error_msg}"
        )


# ============================================================
# 3. CUSTOM CSS
# ============================================================

CSS = """

/* ===== GLOBAL ===== */

:root {
    --bg: #090b12;
    --surface: #111520;
    --surface2: #171d2b;
    --border: #252d3d;
    --text: #f1f5f9;
    --muted: #8994a8;
    --accent: #8b7cff;
    --accent2: #6d5dfc;
}

.gradio-container {
    max-width: 1440px !important;
    margin: auto !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

body {
    background: var(--bg) !important;
}

footer {
    display: none !important;
}


/* ===== HERO ===== */

#hero {
    padding: 42px 20px 30px;
    text-align: center;
}

#hero .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border: 1px solid #343052;
    border-radius: 30px;
    background: #19172c;
    color: #b9adff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

#hero h1 {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    letter-spacing: -1.8px;
    line-height: 1.15;
    margin: 0;
    color: #f8fafc;
}

#hero h1 span {
    background: linear-gradient(
        100deg,
        #a99cff,
        #7865ff,
        #c084fc
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

#hero p {
    color: var(--muted);
    font-size: 14px;
    margin: 13px auto 0;
    max-width: 570px;
    line-height: 1.7;
}


/* ===== PIPELINE ===== */

#pipeline {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 23px;
}

#pipeline .step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 15px;
    background: #151a28;
    border: 1px solid var(--border);
    border-radius: 9px;
    color: #d4d8e4;
    font-size: 12px;
    font-weight: 600;
}

#pipeline .number {
    color: #a99cff;
    font-weight: 800;
}

#pipeline .arrow {
    color: #6659a8;
    font-size: 17px;
}


/* ===== MAIN LAYOUT ===== */

#workspace {
    gap: 20px;
    align-items: stretch;
    margin-top: 10px;
}

.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 22px !important;
    min-width: 0 !important;
}

.panel-title {
    color: #f1f5f9;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 20px;
    line-height: 1.5;
}


/* ===== INPUTS ===== */

.panel label,
.panel .wrap label {
    color: #dce2ed !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

.panel textarea,
.panel input {
    background: #0c111b !important;
    border: 1px solid #30394b !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

.panel textarea:focus,
.panel input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(139, 124, 255, .12) !important;
}

.panel textarea::placeholder,
.panel input::placeholder {
    color: #657086 !important;
}

.panel .image-container,
.panel .upload-container {
    background: #0c111b !important;
    border: 1px dashed #38425a !important;
    border-radius: 12px !important;
}


/* ===== BUTTONS ===== */

#analyze-btn,
#analyze-btn button {
    background: linear-gradient(
        110deg,
        #7564f7,
        #8e65ed
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 46px !important;
    transition: all .2s ease !important;
}

#analyze-btn button:hover {
    filter: brightness(1.12);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(117, 100, 247, .2);
}

#clear-btn,
#clear-btn button {
    background: #1b2231 !important;
    color: #b5bfd0 !important;
    border: 1px solid #30394b !important;
    border-radius: 10px !important;
    min-height: 46px !important;
}


/* ===== EXAMPLES ===== */

.panel .examples-holder {
    margin-top: 15px !important;
}

.panel .examples-holder button {
    background: #171d2b !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: #c5cce0 !important;
}


/* ===== OUTPUT CARDS ===== */

.output-card {
    background: #0c111b;
    border: 1px solid #273044;
    border-radius: 12px;
    padding: 17px;
    margin-bottom: 16px;
    min-height: 155px;
}

.output-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 13px;
}

.output-number {
    width: 27px;
    height: 27px;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 8px;
    background: #282347;
    color: #b9adff;
    font-size: 12px;
    font-weight: 800;
}

.output-name {
    color: #e8eaf2;
    font-size: 13px;
    font-weight: 700;
}

.output-tag {
    margin-left: auto;
    color: #a99cff;
    background: #201d37;
    padding: 4px 8px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .5px;
}


/* ===== MARKDOWN OUTPUT ===== */

.output-card .prose,
.output-card p,
.output-card li {
    color: #c3cada !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
}

.output-card h1,
.output-card h2,
.output-card h3 {
    color: #f1f5f9 !important;
}

.output-card strong {
    color: #e7e4ff !important;
}

.output-card table {
    border-color: #30394b !important;
}

.output-card th {
    background: #171d2b !important;
    color: #dcd7ff !important;
}

.output-card td {
    color: #c3cada !important;
    border-color: #30394b !important;
}


/* ===== FOOTER ===== */

#footer {
    text-align: center;
    padding: 25px 10px 20px;
    margin-top: 28px;
    border-top: 1px solid #222a38;
    color: #778197;
    font-size: 12px;
}

#footer a {
    color: #a99cff;
    text-decoration: none;
    transition: color .2s ease;
}

#footer a:hover {
    color: #d0c8ff;
}


/* ===== RESPONSIVE ===== */

@media (max-width: 768px) {

    #hero {
        padding: 28px 10px 20px;
    }

    #workspace {
        gap: 14px;
    }

    .panel {
        padding: 15px !important;
    }

    #pipeline {
        gap: 7px;
    }

    #pipeline .step {
        font-size: 10px;
        padding: 7px 10px;
    }
}

"""


# ============================================================
# 4. GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Inter"),
            "sans-serif"
        ]
    ),
    css=CSS,
    title="Visual Intelligence | AI Scene Analysis"
) as demo:


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML("""
    <div id="hero">

        <div class="eyebrow">
            <span>●</span>
            AI POWERED VISUAL INTELLIGENCE
        </div>

        <h1>
            See beyond the <span>image.</span>
        </h1>

        <p>
            Upload a scene, ask a question, and get grounded answers
            powered by spatial object detection and multimodal AI reasoning.
        </p>

        <div id="pipeline">

            <div class="step">
                <span class="number">01</span>
                YOLOv8 Detection
            </div>

            <span class="arrow">→</span>

            <div class="step">
                <span class="number">02</span>
                Gemini Reasoning
            </div>

            <span class="arrow">→</span>

            <div class="step">
                <span class="number">03</span>
                Final Answer
            </div>

        </div>

    </div>
    """)


    # --------------------------------------------------------
    # MAIN WORKSPACE
    # --------------------------------------------------------

    with gr.Row(elem_id="workspace"):


        # ====================================================
        # LEFT PANEL: INPUT
        # ====================================================

        with gr.Column(
            scale=5,
            elem_classes=["panel"]
        ):

            gr.HTML("""
            <div class="panel-title">
                Scene Input
            </div>

            <div class="panel-subtitle">
                Upload an image and describe what you want to know.
            </div>
            """)


            input_image = gr.Image(
                type="pil",
                label="Upload your image",
                interactive=True,
                height=310
            )


            input_question = gr.Textbox(
                label="Your question",
                placeholder="e.g. How many people are standing in the top-left corner?",
                lines=3,
                max_lines=5
            )


            with gr.Row():

                submit_btn = gr.Button(
                    "Analyze Scene  →",
                    variant="primary",
                    elem_id="analyze-btn",
                    scale=4
                )

                clear_btn = gr.ClearButton(
                    [
                        input_image,
                        input_question
                    ],
                    value="Clear",
                    elem_id="clear-btn",
                    scale=1
                )


            gr.HTML("""
            <div style="
                margin-top: 22px;
                padding-top: 15px;
                border-top: 1px solid #252d3d;
                color: #8994a8;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: .8px;
            ">
                TRY AN EXAMPLE
            </div>
            """)


            gr.Examples(
                examples=[
                    [
                        "https://raw.githubusercontent.com/MohammedThaher01/multimodal-qa/main/test_images/pexels-bruno-brandao-727244277-18880069.jpg",
                        "How many people are in this scene?"
                    ],
                    [
                        "https://raw.githubusercontent.com/MohammedThaher01/multimodal-qa/main/test_images/noname_13-traffic-3612474_1920.jpg",
                        "How many vehicles are in the top-left?"
                    ],
                ],
                inputs=[
                    input_image,
                    input_question
                ],
                label="Sample images and questions"
            )


        # ====================================================
        # RIGHT PANEL: OUTPUT
        # ====================================================

        with gr.Column(
            scale=7,
            elem_classes=["panel"]
        ):

            gr.HTML("""
            <div class="panel-title">
                Analysis Results
            </div>

            <div class="panel-subtitle">
                Follow the processing pipeline and inspect the
                results from each stage.
            </div>
            """)


            # ------------------------------------------------
            # STAGE 1: SPATIAL METADATA
            # ------------------------------------------------

            gr.HTML("""
            <div class="output-card">

                <div class="output-heading">

                    <div class="output-number">
                        01
                    </div>

                    <div class="output-name">
                        Spatial Metadata
                    </div>

                    <div class="output-tag">
                        YOLOv8
                    </div>

                </div>
            """)


            output_metadata = gr.Markdown(
                value="*Object detection results will appear here.*"
            )


            gr.HTML("</div>")


            # ------------------------------------------------
            # PIPELINE SEPARATOR
            # ------------------------------------------------

            gr.HTML("""
            <div style="
                display:flex;
                align-items:center;
                gap:12px;
                margin: 20px 0;
            ">

                <div style="
                    height:1px;
                    background:#293246;
                    flex:1;
                "></div>

                <span style="
                    color:#78849b;
                    font-size:10px;
                    font-weight:700;
                    letter-spacing:1px;
                ">
                    REASONING LAYER
                </span>

                <div style="
                    height:1px;
                    background:#293246;
                    flex:1;
                "></div>

            </div>
            """)


            # ------------------------------------------------
            # STAGE 2: GEMINI REASONING
            # ------------------------------------------------

            gr.HTML("""
            <div class="output-card">

                <div class="output-heading">

                    <div class="output-number">
                        02
                    </div>

                    <div class="output-name">
                        AI Reasoning
                    </div>

                    <div class="output-tag">
                        GEMINI
                    </div>

                </div>
            """)


            output_answer = gr.Markdown(
                value="*Your AI-generated answer will appear here.*"
            )


            gr.HTML("</div>")


            gr.HTML("""
            <div style="
                color:#778197;
                font-size:11px;
                line-height:1.6;
                margin-top:5px;
            ">
                Results are generated using spatial metadata
                and multimodal image reasoning.
            </div>
            """)


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div id="footer">

        <p>
            Developed by
            <a
                href="https://mohammed-thaher-s.modulify.website/"
                target="_blank"
                rel="noopener noreferrer"
                title="Visit Mohammed Thaher's Portfolio"
                style="
                    color: #c4baff;
                    font-weight: 700;
                    text-decoration: none;
                    transition: all 0.2s ease;
                "
                onmouseover="this.style.color='#ffffff'; this.style.textDecoration='underline';"
                onmouseout="this.style.color='#c4baff'; this.style.textDecoration='none';"
            >
                Mohammed Thaher S
            </a>
        </p>

            <a
                href="https://github.com/MohammedThaher01"
                target="_blank"
                rel="noreferrer"
            >
                GitHub
            </a>

            <a
                href="https://www.linkedin.com/in/mohammed-thaher-s/"
                target="_blank"
                rel="noreferrer"
            >
                LinkedIn
            </a>

        </div>

        <p style="
            margin-top:14px;
            opacity:.6;
        ">
            Grounded Multimodal Visual Q&A
        </p>

    </div>
    """)


    # ========================================================
    # EVENT HANDLER
    # ========================================================

    submit_btn.click(
        fn=process_qa,
        inputs=[
            input_image,
            input_question
        ],
        outputs=[
            output_metadata,
            output_answer
        ],
        show_progress="full"
    )


# ============================================================
# 5. LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
