from flask import Flask, request, render_template, jsonify
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
import traceback

load_dotenv()
app = Flask(__name__)

# Azure Inference API settings
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4o-mini"
token = os.getenv("GITHUB_TOKEN")

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
    news_text = original_news.replace("\n\n", "\n\n[PARAGRAPH_BREAK]\n\n")

    prompt = f"""Original news:
{news_text}

Rewritten news:
"""

    try:
        response = client.complete(
            messages=[
                SystemMessage("You are a professional Nepali news editor. Generate a short and relevant headline, then rewrite the article using standard journalistic Nepali in the same paragraph count."),
                UserMessage(prompt),
            ],
            temperature=0.7,
            top_p=1,
            model=model
        )

        raw_output = response.choices[0].message.content.strip()

        # Parse and format output
        if "**Headline:**" in raw_output and "**Rewritten News:**" in raw_output:
            headline, rewritten_news = raw_output.split("**Rewritten News:**", 1)
            headline = headline.replace("**Headline:**", "").strip()
            paragraphs = [p.strip() for p in rewritten_news.strip().split('\n') if p.strip()]
            formatted_news = '\n\n'.join(paragraphs)

            # Final formatted output with bold headline and spacing
            full_output = f"**{headline}**\n\n{formatted_news}"
            return jsonify({"rewritten_news": full_output})

        # Fallback if formatting is not as expected
        return jsonify({"rewritten_news": raw_output})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
