from flask import Flask, request, jsonify
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Sample FAQs
faq_data = {
    "What is your return policy?": "You can return any product within 30 days of purchase.",
    "How long does shipping take?": "Shipping typically takes 3 to 5 business days.",
    "Do you offer international shipping?": "Yes, we offer international shipping to select countries.",
    "How can I track my order?": "After placing an order, you’ll receive an email with tracking details.",
    "Can I cancel or change my order?": "Orders can be canceled or changed within 24 hours of placement."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Preprocessing: lemmatization & cleaning
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

processed_questions = [preprocess(q) for q in questions]

# Vectorization
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    user_input_clean = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_clean])

    # Cosine similarity
    similarity_scores = cosine_similarity(user_vector, faq_vectors)
    best_match_idx = similarity_scores.argmax()

    response = answers[best_match_idx]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
