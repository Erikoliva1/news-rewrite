from flask import Flask, request, render_template, jsonify
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
import traceback  # ✅ Added for error logging

load_dotenv()

app = Flask(__name__)

# Azure Inference API settings
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4o-mini"
token = os.getenv("GITHUB_TOKEN")

# Check if token is set
if not token:
    raise RuntimeError("GITHUB_TOKEN is not set in environment variables.")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()

    if not data or "news" not in data:
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    original_news = data.get("news", "").strip()

    # Insert paragraph markers
    news_text = original_news.replace("\n\n", "\n\n[PARAGRAPH_BREAK]\n\n")

    prompt = f"""
You are a professional Nepali news editor.

Generate a unique and relevant headline (title) for the following article. Rewrite the following news article in a completely different style and structure with standard literary words & keep same number of paragraphs. Result have to be clearly and professionally by not losing the Journalism standards.

Clearly format the output as:
Headline:
<bold and short topic>

Rewritten News:
<rewritten paragraphs>

Original news:
{news_text}
"""

    try:
        response = client.complete(
            messages=[
                SystemMessage("You are a professional Nepali news editor."),
                UserMessage(prompt),
            ],
            temperature=0.7,
            top_p=1,
            model=model
        )

        # Clean paragraph spacing
        raw_output = response.choices[0].message.content.strip()
        clean_output = '\n'.join([para.strip() for para in raw_output.split('\n') if para.strip()])

        return jsonify({"rewritten_news": clean_output})

    except Exception as e:
        traceback.print_exc()  # ✅ log the actual error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
