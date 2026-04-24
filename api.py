from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

app = Flask(__name__)
CORS(app)

analyzer = SentimentIntensityAnalyzer()

# ✅ Home route (NEW)
@app.route('/')
def home():
    return "✅ Sentiment API is running successfully!"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    comment = data.get("comment", "")

    score = analyzer.polarity_scores(comment)
    compound = score['compound']

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Save to CSV
    with open("user_comments.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([comment, sentiment])

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
