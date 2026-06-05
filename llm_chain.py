import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load the API key from your .env file
load_dotenv()

class VisionQASystem:
    def __init__(self):
        # Load and clean the API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            api_key = api_key.strip()
            print(f"DEBUG: API Key detected (starts with: {api_key[:4]}...)")
        else:
            print("ERROR: GOOGLE_API_KEY not found in environment variables!")
            # We don't raise an error here so the UI can still load and show a helpful message
        
        # Using gemini-3.5-flash which is verified to be available for this API key
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash", 
                google_api_key=api_key
            )
        except Exception as e:
            print(f"ERROR initializing Gemini: {str(e)}")
            self.llm = None


    def _encode_image(self, pil_img):
        """APIs can't send raw image files easily, so we convert the image into a base64 text string."""
        buffered = BytesIO()
        # Ensure image is in standard RGB format
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def ask_question(self, image_pil, yolo_metadata, user_question):
        """Builds the prompt and calls the LLM."""
        
        if not self.llm:
            return "⚠️ Error: Gemini LLM not initialized. Please check if GOOGLE_API_KEY is set in Hugging Face Secrets."
        
        # 1. Convert image to base64
        image_data = self._encode_image(image_pil)

        # 2. This is the Prompt Engineering magic. 
        # We give the LLM a persona and force it to respect YOLO's ground-truth data.
        system_instructions = (
            "You are an expert visual analysis AI. "
            "You have been provided with an image AND absolute ground-truth spatial data extracted by a YOLOv8 object detector. "
            "Rule 1: NEVER contradict the YOLOv8 object counts. Treat them as absolute facts.\n"
            "Rule 2: Use the YOLOv8 spatial layout data to understand where things are located before answering.\n\n"
            f"### YOLOv8 RAW DATA ###\n{yolo_metadata}\n\n"
            f"### USER QUESTION ###\n{user_question}"
        )

        # 3. Package it into a LangChain HumanMessage (Text + Image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": system_instructions},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        )

        # 4. Send to Gemini and return the answer
        response = self.llm.invoke([message])
        
        # Handle case where response.content is a list (common in multimodal responses)
        if isinstance(response.content, list):
            # Join all text parts found in the content list
            text_parts = []
            for item in response.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(text_parts)
            
        return str(response.content)
