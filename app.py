from flask import Flask, render_template, request, jsonify
import requests
import time  

app = Flask(__name__)


API_KEY = "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"
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

MODEL_ENDPOINT = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def chatbot_response(user_message, retries=3):
    """Gets a concise response from the Hugging Face API."""
    
    system_prompt = "You are a knowledgeable AI assistant. Answer concisely and accurately."
    
    data = {
        "inputs": f"User: {user_message}\nAssistant:",
        "parameters": {
            "max_new_tokens": 500,  
            "temperature": 0.5,  
            "top_p": 0.8, 
            "do_sample": True,
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(MODEL_ENDPOINT, headers=HEADERS, json=data)
            response_data = response.json()

           
            if response.status_code == 200 and isinstance(response_data, list) and response_data:
                return response_data[0].get("generated_text", "").strip()

        except Exception as e:
            print(f"Error: {e}")

    return "I couldn't process that. Try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    bot_response = chatbot_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
