from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import traceback
from chatbot import Chatbot

app = Flask(__name__)
CORS(app)

# Initialize chatbot
mongo_uri = "mongodb://localhost:27017/"
db_name = "ai"
collection_names = []
bot = Chatbot(mongo_uri, db_name, collection_names)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"response": "⚠ Please provide a message."}), 400

        response = bot.get_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"❌ Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
