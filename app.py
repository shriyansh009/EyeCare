import os
import io
import re
import json
import base64
import requests
from flask import Flask, render_template, request, session, redirect, url_for
from PIL import Image
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o-mini"  # Vision-capable model

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = "any_secret_key"
UPLOAD_FOLDER = "static/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# Medical Vision Prompt
# --------------------------------------------------
PROMPT = """
You are an Expert Ophthalmologist with decades of clinical experience in ocular pathology and diagnostic imaging.
Analyze the provided eye image and identify abnormalities, infections, or degenerative conditions.

Reply STRICTLY in this JSON format ONLY:
{
  "diagnosis": "string",
  "description": "string",
  "symptoms": "string",
  "home_remedies": ["string1", "string2"],
  "medicines": ["string1", "string2"],
  "disclaimer": "string"
}

Do NOT include markdown, explanations, or text outside the JSON object.
"""

# --------------------------------------------------
# Utility: Extract JSON safely
# --------------------------------------------------
def extract_json(text):
    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception:
        raise ValueError("Invalid JSON returned by OpenRouter model")

# --------------------------------------------------
# OpenRouter Vision Call
# --------------------------------------------------
def call_openrouter_vision(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Eye Disease Analyzer"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # -------------------------------
        # Image from upload
        # -------------------------------
        if "image" in request.files and request.files["image"].filename != "":
            image_file = request.files["image"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
            image_file.save(filepath)

        # -------------------------------
        # Image from camera (base64)
        # -------------------------------
        elif "camera_image" in request.form:
            camera_data = request.form["camera_image"]
            header, encoded = camera_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], "captured.jpg")
            with open(filepath, "wb") as f:
                f.write(img_bytes)

        else:
            return redirect(url_for("index"))

        # -------------------------------
        # Load and encode image
        # -------------------------------
        image = Image.open(filepath).convert("RGB")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()

        # -------------------------------
        # OpenRouter Vision Analysis
        # -------------------------------
        ai_response_text = call_openrouter_vision(img_bytes)
        result = extract_json(ai_response_text)

        # -------------------------------
        # Store result
        # -------------------------------
        session["result"] = result
        session["image_url"] = filepath

        return redirect(url_for("result"))

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/result")
def result():
    return render_template(
        "result.html",
        result=session.get("result"),
        image_url=session.get("image_url")
    )

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
