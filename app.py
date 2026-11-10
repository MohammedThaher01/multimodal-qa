
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

        print(
            f"DEBUG: YOLO metadata extracted: "
            f"{str(yolo_metadata)[:50]}..."
        )

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

        # Return content directly.
        # The UI cards already have their own headings.
        styled_yolo = str(yolo_metadata)
        styled_answer = str(final_answer)

        return styled_yolo, styled_answer

    except Exception as e:

        error_msg = f"ERROR during Gemini analysis: {str(e)}"

        print(f"DEBUG: {error_msg}")

        traceback.print_exc()

        return (
            str(yolo_metadata),
            f"### Error during Stage 2\n\n{error_msg}"
        )


# ============================================================
# 3. CUSTOM CSS
# ============================================================

CSS = """

/* =========================================================
   GLOBAL THEME
   ========================================================= */

:root {
    --bg: #0b0f16;
    --surface: #111722;
    --surface-light: #171f2d;
    --surface-input: #151e2c;

    --border: #293449;
    --border-light: #354157;

    --text-primary: #e8edf5;
    --text-secondary: #a3afc1;
    --text-muted: #7d8ba1;

    --accent: #a99aff;
    --accent-strong: #8875ee;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    background: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

body {
    background: var(--bg) !important;
}

footer {
    display: none !important;
}


/* =========================================================
   HERO SECTION
   ========================================================= */

#hero {
    text-align: center;
    padding: 42px 20px 30px;
}

#hero .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 7px 13px;

    background: #191b2b;
    border: 1px solid #36324f;
    border-radius: 30px;

    color: #c0b5ff;

    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;

    margin-bottom: 20px;
}

#hero .eyebrow span {
    color: #9d8cff;
    font-size: 9px;
}

#hero h1 {
    color: #f1f4f9;

    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;

    letter-spacing: -1.8px;
    line-height: 1.15;

    margin: 0;
}

#hero h1 span {
    background: linear-gradient(
        100deg,
        #b5a8ff,
        #8a75f0,
        #c4a6ed
    );

    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

#hero p {
    color: var(--text-secondary);

    font-size: 14px;
    line-height: 1.8;

    max-width: 600px;
    margin: 14px auto 0;
}


/* =========================================================
   PIPELINE INDICATOR
   ========================================================= */

#pipeline {
    display: flex;
    justify-content: center;
    align-items: center;

    gap: 11px;
    flex-wrap: wrap;

    margin-top: 25px;
}

#pipeline .step {
    display: flex;
    align-items: center;
    gap: 9px;

    padding: 9px 15px;

    background: #141b27;
    border: 1px solid var(--border);
    border-radius: 8px;

    color: #cbd5e1;

    font-size: 12px;
    font-weight: 600;
}

#pipeline .number {
    color: #b2a4ff;
    font-weight: 800;
}

#pipeline .arrow {
    color: #6d638f;
    font-size: 16px;
}


/* =========================================================
   MAIN WORKSPACE
   ========================================================= */

#workspace {
    gap: 20px;
    align-items: stretch;
    margin-top: 12px;
}

.panel {
    min-width: 0 !important;

    padding: 22px !important;

    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

.panel-title {
    color: var(--text-primary);

    font-size: 16px;
    font-weight: 700;

    margin-bottom: 6px;
}

.panel-subtitle {
    color: var(--text-secondary);

    font-size: 12px;
    line-height: 1.7;

    margin-bottom: 21px;
}


/* =========================================================
   INPUT LABELS AND CONTROLS
   ========================================================= */

.panel label,
.panel .wrap label {
    color: #dce3ed !important;

    font-size: 13px !important;
    font-weight: 600 !important;
}

.panel textarea,
.panel input {
    background: var(--surface-input) !important;

    border: 1px solid var(--border-light) !important;
    border-radius: 9px !important;

    color: #e8edf5 !important;

    font-size: 14px !important;
}

.panel textarea::placeholder,
.panel input::placeholder {
    color: #8290a5 !important;
}

.panel textarea:focus,
.panel input:focus {
    border-color: #9583f5 !important;

    box-shadow: 0 0 0 2px
        rgba(149, 131, 245, 0.13) !important;
}

/* Image upload area */

.panel .image-container,
.panel .upload-container {
    background: #101722 !important;

    border: 1px dashed #46516a !important;
    border-radius: 10px !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

#analyze-btn,
#analyze-btn button {
    min-height: 46px !important;

    background: linear-gradient(
        110deg,
        #7866e8,
        #9270e9
    ) !important;

    color: #ffffff !important;

    border: none !important;
    border-radius: 9px !important;

    font-weight: 700 !important;

    transition: all 0.2s ease !important;
}

#analyze-btn button:hover {
    filter: brightness(1.12);

    transform: translateY(-1px);

    box-shadow: 0 5px 18px
        rgba(120, 102, 232, 0.2);
}

#clear-btn,
#clear-btn button {
    min-height: 46px !important;

    background: #1b2433 !important;

    color: #c3cddd !important;

    border: 1px solid #354157 !important;
    border-radius: 9px !important;

    font-weight: 600 !important;
}

#clear-btn button:hover {
    background: #253044 !important;
    color: #f1f5f9 !important;
}


/* =========================================================
   EXAMPLES
   ========================================================= */

.panel .examples-holder {
    margin-top: 15px !important;
}

.panel .examples-holder button {
    background: #151e2c !important;

    border: 1px solid var(--border) !important;
    border-radius: 8px !important;

    color: #cbd5e1 !important;
}

.panel .examples-holder button:hover {
    border-color: #7266b4 !important;
}


/* =========================================================
   OUTPUT CARDS
   IMPORTANT:
   The output Markdown components are actual children
   of these cards through gr.Group.
   ========================================================= */

.output-card {
    min-height: 0 !important;

    padding: 20px !important;
    margin-bottom: 16px !important;

    background: #111722 !important;

    border: 1px solid #2b364b !important;
    border-radius: 12px !important;

    box-shadow: 0 4px 14px
        rgba(0, 0, 0, 0.12) !important;
}


/* =========================================================
   OUTPUT CARD HEADERS
   ========================================================= */

.output-heading {
    display: flex;
    align-items: center;

    gap: 11px;

    padding-bottom: 15px;
    margin-bottom: 15px;

    border-bottom: 1px solid #293449;
}

.output-number {
    width: 30px;
    height: 30px;

    display: flex;
    justify-content: center;
    align-items: center;

    flex-shrink: 0;

    background: #292541;
    border: 1px solid #3b365c;
    border-radius: 7px;

    color: #c0b5ff;

    font-size: 12px;
    font-weight: 800;
}

.output-name {
    color: #e5eaf2;

    font-size: 14px;
    font-weight: 700;
}

.output-tag {
    margin-left: auto;

    padding: 5px 9px;

    background: #211f34;
    border: 1px solid #38324f;
    border-radius: 6px;

    color: #b9adff;

    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
}


/* =========================================================
   MARKDOWN RESULTS
   ========================================================= */

.output-card .prose,
.output-card .markdown,
.output-card p,
.output-card li {
    color: #cbd5e1 !important;

    font-size: 14px !important;
    line-height: 1.85 !important;
}

.output-card h1,
.output-card h2,
.output-card h3,
.output-card h4 {
    color: #f0f3f8 !important;

    font-weight: 650 !important;

    margin-top: 15px !important;
    margin-bottom: 9px !important;
}

.output-card h1 {
    font-size: 21px !important;
}

.output-card h2 {
    font-size: 18px !important;
}

.output-card h3 {
    font-size: 15px !important;
}

.output-card strong {
    color: #f1f4f9 !important;
    font-weight: 700 !important;
}

.output-card em {
    color: #aab5c5 !important;
}

.output-card ul,
.output-card ol {
    padding-left: 22px !important;
    margin: 9px 0 !important;
}

.output-card li {
    padding-left: 4px !important;
    margin-bottom: 5px !important;
}


/* Inline code */

.output-card code {
    background: #202a3a !important;
    color: #c4b5fd !important;

    padding: 2px 5px;
    border-radius: 4px;
}


/* Code blocks */

.output-card pre {
    background: #0c121c !important;

    border: 1px solid #293449 !important;
    border-radius: 8px !important;

    padding: 13px !important;
}


/* Tables */

.output-card table {
    width: 100% !important;

    border-collapse: collapse !important;

    border: 1px solid #303b50 !important;
    border-radius: 8px !important;
}

.output-card th {
    background: #1b2433 !important;

    color: #e5eaf2 !important;
    font-weight: 650 !important;
}

.output-card td {
    background: #111722 !important;

    color: #cbd5e1 !important;
    border-color: #303b50 !important;
}

.output-card th,
.output-card td {
    padding: 10px 12px !important;
}

.output-card a {
    color: #b3a5ff !important;
}


/* =========================================================
   REASONING DIVIDER
   ========================================================= */

.reasoning-divider {
    display: flex;
    align-items: center;

    gap: 14px;

    margin: 22px 0;
}

.reasoning-divider::before,
.reasoning-divider::after {
    content: "";

    flex: 1;

    height: 1px;

    background: #293449;
}

.reasoning-divider span {
    color: #8995a8;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 1.2px;
}


/* =========================================================
   RESULTS FOOTNOTE
   ========================================================= */

.results-note {
    color: #8995a8;

    font-size: 12px;
    line-height: 1.7;

    margin-top: 8px;
}


/* =========================================================
   FOOTER
   ========================================================= */

#footer {
    text-align: center;

    padding: 25px 10px 20px;
    margin-top: 28px;

    border-top: 1px solid #293449;

    color: #8b98ab;
    font-size: 12px;
}

#footer a {
    color: #a99aff;

    text-decoration: none;

    transition: color 0.2s ease;
}

#footer a:hover {
    color: #e0d9ff;
}

#footer .developer-link {
    color: #c5baff;

    font-weight: 700;

    text-decoration: none;

    transition: all 0.2s ease;
}

#footer .developer-link:hover {
    color: #ffffff;
    text-decoration: underline;
}


/* =========================================================
   RESPONSIVE DESIGN
   ========================================================= */

@media (max-width: 768px) {

    #hero {
        padding: 30px 12px 22px;
    }

    #hero h1 {
        font-size: 32px;
        letter-spacing: -1px;
    }

    #workspace {
        gap: 14px;
    }

    .panel {
        padding: 16px !important;
    }

    #pipeline {
        gap: 7px;
    }

    #pipeline .step {
        padding: 7px 10px;
        font-size: 10px;
    }

    .output-card {
        padding: 15px !important;
    }

    .output-card p,
    .output-card li {
        font-size: 13px !important;
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


    # ========================================================
    # HEADER
    # ========================================================

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


    # ========================================================
    # MAIN WORKSPACE
    # ========================================================

    with gr.Row(elem_id="workspace"):


        # ----------------------------------------------------
        # LEFT PANEL: INPUT
        # ----------------------------------------------------

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


            # Examples separator

            gr.HTML("""
            <div style="
                margin-top: 22px;
                padding-top: 15px;
                border-top: 1px solid #293449;
                color: #8995a8;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            ">
                TRY AN EXAMPLE
            </div>
            """)


            # Example images

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


        # ----------------------------------------------------
        # RIGHT PANEL: RESULTS
        # ----------------------------------------------------

        with gr.Column(
            scale=7,
            elem_classes=["panel"]
        ):

            gr.HTML("""
            <div class="panel-title">
                Analysis Results
            </div>

            <div class="panel-subtitle">
                Spatial detection and AI-generated insights from your image.
            </div>
            """)


            # =================================================
            # STAGE 1: SPATIAL METADATA
            # =================================================

            with gr.Group(
                elem_classes=["output-card"]
            ):

                gr.HTML("""
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
                    value=(
                        "*Spatial detection results will appear "
                        "here after analysis.*"
                    )
                )


            # =================================================
            # REASONING DIVIDER
            # =================================================

            gr.HTML("""
            <div class="reasoning-divider">
                <span>REASONING LAYER</span>
            </div>
            """)


            # =================================================
            # STAGE 2: GEMINI REASONING
            # =================================================

            with gr.Group(
                elem_classes=["output-card"]
            ):

                gr.HTML("""
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
                    value=(
                        "*Your AI-generated answer will appear "
                        "here after analysis.*"
                    )
                )


            # =================================================
            # RESULTS FOOTNOTE
            # =================================================

            gr.HTML("""
            <div class="results-note">
                Analysis combines YOLOv8 spatial detection
                with Gemini multimodal reasoning.
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
                class="developer-link"
                href="https://mohammed-thaher-s.modulify.website/"
                target="_blank"
                rel="noopener noreferrer"
                title="Visit Mohammed Thaher's Portfolio"
            >
                Mohammed Thaher S
            </a>
        </p>

        <div style="
            display: flex;
            justify-content: center;
            gap: 22px;
            margin-top: 12px;
        ">

            <a
                href="https://github.com/MohammedThaher01"
                target="_blank"
                rel="noopener noreferrer"
            >
                GitHub
            </a>

            <a
                href="https://www.linkedin.com/in/mohammed-thaher-s/"
                target="_blank"
                rel="noopener noreferrer"
            >
                LinkedIn
            </a>

        </div>

        <p style="
            margin-top: 15px;
            color: #69778c;
            font-size: 11px;
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
# 5. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
