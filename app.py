
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
        # Stage 1: YOLOv8 Spatial Analysis
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
        # Stage 2: Gemini Multimodal Reasoning
        progress(0.6, desc="Stage 2: Gemini Multimodal Reasoning...")

        print("DEBUG: Starting Gemini analysis...")

        final_answer = qa_system.ask_question(
            image,
            yolo_metadata,
            question
        )

        print("DEBUG: Gemini analysis complete.")

        progress(1.0, desc="Analysis Complete!")

        return str(yolo_metadata), str(final_answer)

    except Exception as e:

        error_msg = f"ERROR during Gemini analysis: {str(e)}"

        print(f"DEBUG: {error_msg}")

        traceback.print_exc()

        return (
            str(yolo_metadata),
            f"### Error during Stage 2\n\n{error_msg}"
        )


# ============================================================
# 3. RED THEME CSS
# ============================================================

CSS = """

/* =========================================================
   COLOR SYSTEM
   ========================================================= */

:root {
    --bg: #0b0d11;
    --surface: #12161e;
    --surface-light: #191e28;
    --surface-input: #171c26;

    --border: #2b303b;
    --border-light: #3a414f;

    --text-primary: #f1f2f4;
    --text-secondary: #b0b7c3;
    --text-muted: #858e9d;

    --accent: #ef4444;
    --accent-light: #ff6565;
    --accent-dark: #b91c1c;

    --accent-bg: #351719;
    --accent-border: #69282b;
}


/* =========================================================
   GLOBAL
   ========================================================= */

body {
    background: var(--bg) !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;

    background: var(--bg) !important;
    color: var(--text-primary) !important;

    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

.gradio-container,
.gradio-container * {
    scrollbar-color: #444c59 #11151c;
}

footer {
    display: none !important;
}


/* =========================================================
   HERO
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

    background: #241719 !important;
    border: 1px solid #633036 !important;
    border-radius: 30px;

    color: #ff8585 !important;

    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;

    margin-bottom: 20px;
}

#hero .eyebrow span {
    color: #f05252 !important;
    font-size: 9px;
}

#hero h1 {
    color: #f5f5f5 !important;

    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;

    letter-spacing: -1.8px;
    line-height: 1.15;

    margin: 0;
}

#hero h1 span {
    background: linear-gradient(
        100deg,
        #ff7777,
        #ef4444,
        #bd2525
    ) !important;

    -webkit-background-clip: text !important;
    background-clip: text !important;

    color: transparent !important;
}

#hero p {
    color: #aeb6c3 !important;

    font-size: 14px;
    line-height: 1.8;

    max-width: 600px;
    margin: 14px auto 0;
}


/* =========================================================
   PIPELINE
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

    background: #171a22 !important;
    border: 1px solid #333743 !important;
    border-radius: 8px;

    color: #d3d7de !important;

    font-size: 12px;
    font-weight: 600;
}

#pipeline .number {
    color: #ff7474 !important;
    font-weight: 800;
}

#pipeline .arrow {
    color: #d34c4c !important;
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

    background: #12161e !important;
    border: 1px solid #2b303b !important;
    border-radius: 14px !important;
}

.panel-title {
    color: #f0f1f3 !important;

    font-size: 16px;
    font-weight: 700;

    margin-bottom: 6px;
}

.panel-subtitle {
    color: #aeb6c3 !important;

    font-size: 12px;
    line-height: 1.7;

    margin-bottom: 21px;
}


/* =========================================================
   INPUTS
   ========================================================= */

.panel label,
.panel .wrap label {
    color: #e1e4e9 !important;

    font-size: 13px !important;
    font-weight: 600 !important;
}

.panel textarea,
.panel input {
    background: #171c26 !important;

    border: 1px solid #39404d !important;
    border-radius: 9px !important;

    color: #f0f2f5 !important;

    font-size: 14px !important;
}

.panel textarea::placeholder,
.panel input::placeholder {
    color: #8993a2 !important;
}

.panel textarea:focus,
.panel input:focus {
    border-color: #ef4444 !important;

    box-shadow: 0 0 0 2px
        rgba(239, 68, 68, 0.16) !important;
}

.panel .image-container,
.panel .upload-container {
    background: #151a23 !important;

    border: 1px dashed #55515a !important;
    border-radius: 10px !important;
}


/* =========================================================
   PRIMARY BUTTON
   ========================================================= */

#analyze-btn,
#analyze-btn button {
    min-height: 46px !important;

    background: linear-gradient(
        110deg,
        #c92d35,
        #ef4444
    ) !important;

    color: #ffffff !important;

    border: 1px solid #ef4444 !important;
    border-radius: 9px !important;

    font-weight: 700 !important;

    transition: all 0.2s ease !important;
}

#analyze-btn button:hover {
    background: linear-gradient(
        110deg,
        #df343c,
        #ff5959
    ) !important;

    border-color: #ff6969 !important;

    transform: translateY(-1px);

    box-shadow: 0 5px 18px
        rgba(239, 68, 68, 0.22) !important;
}


/* =========================================================
   CLEAR BUTTON
   ========================================================= */

#clear-btn,
#clear-btn button {
    min-height: 46px !important;

    background: #1d222c !important;

    color: #c5cbd5 !important;

    border: 1px solid #3a414f !important;
    border-radius: 9px !important;

    font-weight: 600 !important;
}

#clear-btn button:hover {
    background: #292e39 !important;
    color: #ffffff !important;
    border-color: #555d6b !important;
}


/* =========================================================
   EXAMPLES
   ========================================================= */

.panel .examples-holder {
    margin-top: 15px !important;
}

.panel .examples-holder button {
    background: #171c26 !important;

    border: 1px solid #343b47 !important;
    border-radius: 8px !important;

    color: #d4d9e1 !important;
}

.panel .examples-holder button:hover {
    border-color: #a53b41 !important;
    background: #211a20 !important;
}


/* =========================================================
   OUTPUT CARDS
   ========================================================= */

.output-card {
    min-height: 0 !important;

    padding: 20px !important;
    margin-bottom: 16px !important;

    background: #141922 !important;

    border: 1px solid #343b48 !important;
    border-radius: 12px !important;

    box-shadow: 0 4px 14px
        rgba(0, 0, 0, 0.15) !important;
}


/* =========================================================
   OUTPUT HEADINGS
   ========================================================= */

.output-heading {
    display: flex;
    align-items: center;

    gap: 11px;

    padding-bottom: 15px;
    margin-bottom: 15px;

    border-bottom: 1px solid #343b47;
}

.output-number {
    width: 30px;
    height: 30px;

    display: flex;
    justify-content: center;
    align-items: center;

    flex-shrink: 0;

    background: #371b20 !important;
    border: 1px solid #713038 !important;
    border-radius: 7px;

    color: #ff8989 !important;

    font-size: 12px;
    font-weight: 800;
}

.output-name {
    color: #eceef2 !important;

    font-size: 14px;
    font-weight: 700;
}

.output-tag {
    margin-left: auto;

    padding: 5px 9px;

    background: #321a1d !important;
    border: 1px solid #693036 !important;
    border-radius: 6px;

    color: #ff9999 !important;

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
    color: #cbd1da !important;

    font-size: 14px !important;
    line-height: 1.85 !important;
}

.output-card h1,
.output-card h2,
.output-card h3,
.output-card h4 {
    color: #f2f3f5 !important;

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
    color: #ffffff !important;
    font-weight: 700 !important;
}

.output-card em {
    color: #aeb7c5 !important;
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


/* =========================================================
   INLINE CODE
   ========================================================= */

.output-card code {
    background: #281b20 !important;
    color: #ff9999 !important;

    padding: 2px 5px;
    border-radius: 4px;
}


/* =========================================================
   CODE BLOCKS
   ========================================================= */

.output-card pre {
    background: #0d1118 !important;

    border: 1px solid #373d48 !important;
    border-radius: 8px !important;

    padding: 13px !important;
}


/* =========================================================
   TABLES
   ========================================================= */

.output-card table {
    width: 100% !important;

    border-collapse: collapse !important;

    border: 1px solid #393f4b !important;
    border-radius: 8px !important;
}

.output-card th {
    background: #242a34 !important;

    color: #f0f1f3 !important;
    font-weight: 650 !important;
}

.output-card td {
    background: #141922 !important;

    color: #cbd1da !important;
    border-color: #393f4b !important;
}

.output-card th,
.output-card td {
    padding: 10px 12px !important;
}

.output-card a {
    color: #ff8585 !important;
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

    background: #343b47;
}

.reasoning-divider span {
    color: #a0a9b7 !important;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 1.2px;
}


/* =========================================================
   RESULTS FOOTNOTE
   ========================================================= */

.results-note {
    color: #929dac !important;

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

    border-top: 1px solid #303640;

    color: #a0a9b7 !important;
    font-size: 12px;
}

#footer a {
    color: #ff8585 !important;

    text-decoration: none;

    transition: color 0.2s ease;
}

#footer a:hover {
    color: #ffffff !important;
}

#footer .developer-link {
    color: #ff8585 !important;

    font-weight: 700;

    text-decoration: none;

    transition: all 0.2s ease;
}

#footer .developer-link:hover {
    color: #ffffff !important;
    text-decoration: underline;
}


/* =========================================================
   RESPONSIVE
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
        primary_hue="red",
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
        # LEFT: INPUT PANEL
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
                border-top: 1px solid #303640;
                color: #a0a9b7;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
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


        # ----------------------------------------------------
        # RIGHT: RESULTS PANEL
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


            # ------------------------------------------------
            # STAGE 1: YOLO
            # ------------------------------------------------

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


            # ------------------------------------------------
            # REASONING DIVIDER
            # ------------------------------------------------

            gr.HTML("""
            <div class="reasoning-divider">
                <span>REASONING LAYER</span>
            </div>
            """)


            # ------------------------------------------------
            # STAGE 2: GEMINI
            # ------------------------------------------------

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


            # Results footnote

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
            color: #858e9d;
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
# 5. LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
