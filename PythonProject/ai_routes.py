import os
import google.generativeai as genai
from flask import Blueprint, request, jsonify
import traceback

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

ai_bp = Blueprint("ai", __name__)

@ai_bp.route("/api/ask", methods=["POST"])
def ask():
    """
    Ask Gemini AI
    ---
    tags:
      - Gemini AI
    parameters:
      - in: body
        name: prompt
        schema:
          type: object
          required:
            - prompt
          properties:
            prompt:
              type: string
              example: "What is the capital of Japan?"
    responses:
      200:
        description: AI response
        schema:
          type: object
          properties:
            response:
              type: string
      400:
        description: Bad Request
      500:
        description: Internal Server Error
    """
    try:
        prompt = request.json.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return jsonify({"response": response.text.strip()})
        else:
            return jsonify({"error": "Gemini returned no text."})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Exception: {str(e)}"}), 500
