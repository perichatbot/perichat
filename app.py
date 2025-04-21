from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
from chatbot import Chatbot  # Ensure this imports from the updated chatbot class

app = Flask(__name__)
CORS(app)

# ✅ Initialize chatbot without Google Translate
try:
    bot = Chatbot()  # No need for Google Translate support
except Exception as e:
    print(f"❌ Failed to initialize chatbot: {e}")
    bot = None

@app.route("/")
def home():
    return render_template("index.html")  # Ensure templates/index.html exists

@app.route("/chat", methods=["POST"])
def chat():
    if not bot:
        return jsonify({"response": "❌ Chatbot not initialized."}), 500

    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "⚠ Please provide a message."}), 400

        response = bot.get_response(user_input)
        return jsonify({"response": response})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"❌ Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
