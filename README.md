📊 E-Consult Feedback Analysis Dashboard
A powerful and interactive dashboard built using Streamlit that analyzes user feedback and converts it into meaningful insights. This tool helps identify sentiment patterns (positive, negative, neutral) and supports better decision-making through clear visualizations and data analysis.

🚀 Overview

The E-Consult Feedback Analysis Dashboard allows users to upload and analyze feedback data easily. It uses VADER sentiment analysis to classify comments and presents the results through dynamic charts, filters, and summaries. This makes it useful for domains like healthcare, technology, education, and business.

✨ Features
🔍 Sentiment Analysis – Classifies comments into positive, negative, and neutral
📊 Interactive Visualizations – Pie charts, bar graphs, and word clouds
📁 CSV Upload Support – Analyze your own datasets
🎯 Smart Filters – Filter data by sentiment and category
🔎 Keyword Search – Find specific terms in comments
📄 Report Generation – Export insights for further use
💡 Automated Insights – Get suggestions based on analyzed data
🧠 Technology Stack
Frontend & Backend: Streamlit
Sentiment Analysis: VADER
Data Processing: Pandas, NumPy
Visualization: Plotly, Matplotlib, WordCloud
⚙️ How to Run Locally

Clone the repository

git clone <your-repo-link>
cd feedback-analysis-dashboard

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app.py
Open in browser:
http://localhost:8501
📂 Dataset Format

Your CSV file should include:

sentiment,domain,comment
positive,healthcare,Great service
negative,technical,App crashes frequently
neutral,billing,Average experience
📈 Key Insights
Helps identify strengths and weaknesses from feedback
Detects sentiment trends across different domains
Supports data-driven improvements
Easy to use for both technical and non-technical users
📌 Conclusion

This project simplifies feedback analysis by combining sentiment detection with visual storytelling. It is a useful tool for anyone looking to understand user opinions and improve their services based on real data.
