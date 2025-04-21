import os
import pandas as pd
import re
import numpy as np
import faiss
import sympy as sp
import spacy
import requests
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()


class Chatbot:
    def __init__(self, mongo_uri, db_name, collection_names):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        self.model = SentenceTransformer("BAAI/bge-large-en")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection_names = collection_names

        self.vectorizer = CountVectorizer()
        self.data = self.load_data()

        if not self.data.empty:
            self.index, self.embeddings = self.build_faiss_index(self.data['Cleaned_Question'])
            self.bow_matrix = self.vectorizer.fit_transform(self.data['Cleaned_Question'])
        else:
            self.index, self.embeddings, self.bow_matrix = None, None, None

    def load_data(self):
        all_data = []
        for name in self.collection_names:
            try:
                collection = self.db[name]
                documents = list(collection.find({}, {"_id": 0, "Question": 1, "Response": 1}))
                df = pd.DataFrame(documents)
                if {'Question', 'Response'}.issubset(df.columns):
                    df['Cleaned_Question'] = df['Question'].apply(self.clean_text)
                    all_data.append(df)
                    print(f"✅ Loaded {len(df)} entries from '{name}'")
                else:
                    print(f"⚠ Collection '{name}' missing required fields.")
            except Exception as e:
                print(f"❌ Error loading collection '{name}': {e}")
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame(columns=['Question', 'Response', 'Cleaned_Question'])

    def clean_text(self, text):
        doc = self.nlp(str(text))
        return ' '.join(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct)

    def is_math_expression(self, text):
        return bool(re.fullmatch(r'[\d\s\+\-\*/\^\(\)\.]+', text.strip()))

    def solve_math_expression(self, expression):
        try:
            result = sp.sympify(expression).evalf()
            return f"Answer: {result}"
        except Exception:
            return "⚠ Invalid mathematical expression."

    def build_faiss_index(self, clean_questions):
        embeddings = self.model.encode(clean_questions.tolist(), convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"✅ FAISS index built with {len(clean_questions)} entries")
        return index, embeddings

    def get_bow_response(self, user_input):
        cleaned_input = self.clean_text(user_input)
        input_vec = self.vectorizer.transform([cleaned_input])
        cosine_similarities = cosine_similarity(input_vec, self.bow_matrix).flatten()
        best_idx = np.argmax(cosine_similarities)
        best_score = cosine_similarities[best_idx]

        if best_score > 0.2:
            print(f"🧠 BoW Match Score: {round(best_score * 100, 2)}%")
            return self.data.iloc[best_idx]['Response']
        return "❌ Sorry, I don't understand that question."

    def call_groq(self, user_input):
        print("🚀 Calling Groq API...")  # Debug line

        try:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                return "❌ Groq API key not set. Please set the 'GROQ_API_KEY' environment variable."

            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant trained to answer technical and general queries."},
                    {"role": "user", "content": user_input}
                ]
            }
            response = requests.post(url, headers=headers, json=payload)
            print(f"🔁 Groq status code: {response.status_code}")  # Debug info

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"⚠ Groq error {response.status_code}: {response.text}"
        except Exception as e:
            return f"❌ Failed to contact Groq: {e}"

    def get_response(self, user_input):
        if not user_input.strip():
            return "⚠ Please enter a valid question."

        if self.is_math_expression(user_input):
            return self.solve_math_expression(user_input)

        # 🛠 Fix: fallback to Groq if dataset is missing
        if self.index is None or self.data.empty:
            print("⚠ No dataset found. Using Groq directly...")
            return self.call_groq(user_input)

        cleaned_input = self.clean_text(user_input)
        input_embedding = self.model.encode([cleaned_input], convert_to_numpy=True)
        distances, best_match_idx = self.index.search(input_embedding, 3)
        best_match_idx = best_match_idx[0]

        for idx in best_match_idx:
            if idx < len(self.data):
                score = fuzz.ratio(cleaned_input, self.data.iloc[idx]['Cleaned_Question'])
                if score > 50:
                    print(f"🔹 Best Match Score: {score}%")
                    return self.data.iloc[idx]['Response']

        print("⚠ Using BoW fallback...")
        bow_response = self.get_bow_response(user_input)

        if "❌" in bow_response:
            print("⚠ BoW failed. Calling Groq...")
            return self.call_groq(user_input)

        return bow_response

    def chat(self):
        print("🤖 Hello! I am your chatbot. Type 'bye' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower().strip() == 'bye':
                print("Chatbot: Goodbye! 👋")
                break
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")


if __name__ == "__main__":
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "ai"
    collection_names = []  # Can be empty now — Groq will take over

    bot = Chatbot(mongo_uri, db_name, collection_names)
    bot.chat()
