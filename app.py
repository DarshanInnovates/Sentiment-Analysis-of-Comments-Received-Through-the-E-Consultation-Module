import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Setting page configuration
st.set_page_config(page_title="E-Consult Feedback Analysis", layout="wide")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Title and introduction
st.title("E-Consult Feedback Analysis Dashboard 📊")
st.markdown("""
Unleash the power of feedback with the E-Consult Feedback Analysis Dashboard! This versatile tool empowers everyone—businesses, educators, developers, researchers, and individuals—to transform comments into actionable insights across any field, from healthcare and education to technology and creative arts. Upload your own CSV dataset or explore our sample dataset to uncover positive, negative, and neutral sentiment trends across diverse categories. With vibrant visualizations, intuitive filters, and automated sentiment analysis powered by VADER, this dashboard highlights strengths, identifies challenges, and fuels innovation. Hosted on Streamlit, it delivers seamless, data-driven decision-making for all, anywhere, anytime.
""")

# Load default dataset
@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("econsult_comments_dataset.csv")
        df = df.dropna()
        df['sentiment_label'] = df['sentiment_label'].str.lower().str.strip()
        df['domain'] = df['domain'].str.lower().str.strip()
        # Compute VADER scores for default dataset
        df['vader_score'] = df['comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        return df
    except FileNotFoundError:
        st.error("Default dataset 'econsult_comments_dataset.csv' not found in the project directory.")
        return None

# Process uploaded data and predict sentiment
def process_uploaded_data(file):
    try:
        uploaded_df = pd.read_csv(file)
        if 'comment' not in uploaded_df.columns:
            st.error("Uploaded CSV must contain a 'comment' column.")
            return None
        uploaded_df = uploaded_df.dropna(subset=['comment'])
        # Compute VADER scores
        uploaded_df['vader_score'] = uploaded_df['comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        # Predict sentiment if not provided
        if 'sentiment_label' not in uploaded_df.columns:
            uploaded_df['sentiment_label'] = uploaded_df['vader_score'].apply(
                lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
            )
        # Use 'general' as default domain if not provided
        if 'domain' not in uploaded_df.columns:
            uploaded_df['domain'] = 'general'
        uploaded_df['sentiment_label'] = uploaded_df['sentiment_label'].str.lower().str.strip()
        uploaded_df['domain'] = uploaded_df['domain'].str.lower().str.strip()
        return uploaded_df
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None

# Sidebar for data source selection and file upload
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose data source", ["Default Dataset", "Upload Your Dataset"])
uploaded_df = None
if data_source == "Upload Your Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file (must include 'comment' column)", type=["csv"])
    if uploaded_file:
        uploaded_df = process_uploaded_data(uploaded_file)
        if uploaded_df is not None:
            st.sidebar.success("File uploaded successfully! Sentiment analysis applied.")

# Load data based on user choice
if data_source == "Upload Your Dataset" and uploaded_df is not None:
    df = uploaded_df
else:
    df = load_default_data()
    if df is None:
        st.stop()

# Filtering options
st.sidebar.header("Filter Options")
sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    options=df['sentiment_label'].unique(),
    default=df['sentiment_label'].unique()
)
domain_filter = st.sidebar.multiselect(
    "Select Domain",
    options=df['domain'].unique(),
    default=df['domain'].unique()
)

# Keyword search
st.sidebar.header("Keyword Search")
keyword = st.sidebar.text_input("Enter keyword to search comments")

# Apply filters
filtered_df = df[
    (df['sentiment_label'].isin(sentiment_filter)) &
    (df['domain'].isin(domain_filter))
]
if keyword:
    filtered_df = filtered_df[filtered_df['comment'].str.contains(keyword, case=False, na=False)]

# Summary section
st.header("Summary")
if filtered_df.empty:
    st.warning("No data matches the selected filters or keyword. Please adjust the filters.")
else:
    total_comments = len(filtered_df)
    positive_comments = len(filtered_df[filtered_df['sentiment_label'] == 'positive'])
    negative_comments = len(filtered_df[filtered_df['sentiment_label'] == 'negative'])
    neutral_comments = len(filtered_df[filtered_df['sentiment_label'] == 'neutral'])
    pos_percentage = (positive_comments / total_comments * 100) if total_comments > 0 else 0
    neg_percentage = (negative_comments / total_comments * 100) if total_comments > 0 else 0
    neu_percentage = (neutral_comments / total_comments * 100) if total_comments > 0 else 0
    st.write(f"""
    - **Data Source**: {data_source}
    - **Total Comments**: {total_comments}
    - **Positive Comments**: {positive_comments} ({pos_percentage:.1f}%)
    - **Negative Comments**: {negative_comments} ({neg_percentage:.1f}%)
    - **Neutral Comments**: {neutral_comments} ({neu_percentage:.1f}%)
    """)

# Visualization: Sentiment Distribution
st.header("Sentiment Distribution")
if not filtered_df.empty:
    sentiment_counts = filtered_df['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig1 = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        title="Sentiment Distribution",
        color='Sentiment',
        color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B', 'neutral': '#636EFA'}
    )
    fig1.update_traces(textinfo='percent+label')
    fig1.update_layout(showlegend=True)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("No data to display for sentiment distribution.")

# Visualization: Domain-wise Sentiment
st.header("Domain-wise Sentiment Analysis")
if not filtered_df.empty:
    try:
        domain_sentiment = filtered_df.groupby(['domain', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
        available_sentiments = [col for col in domain_sentiment.columns if col in ['positive', 'negative', 'neutral']]
        if not available_sentiments:
            st.warning("No sentiment data available for domain-wise analysis.")
        else:
            domain_sentiment = domain_sentiment.melt(id_vars='domain', value_vars=available_sentiments, var_name='Sentiment', value_name='Count')
            fig2 = px.bar(
                domain_sentiment,
                x='domain',
                y='Count',
                color='Sentiment',
                barmode='stack',
                title="Sentiment by Domain",
                color_discrete_map={'positive': '#00CC96', 'negative': '#EF553B', 'neutral': '#636EFA'}
            )
            fig2.update_layout(
                xaxis_title="Domain",
                yaxis_title="Number of Comments",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating domain-wise sentiment chart: {str(e)}")
else:
    st.warning("No data to display for domain-wise sentiment analysis.")

# Visualization: Word Cloud
st.header("Comment Word Cloud")
if not filtered_df.empty:
    try:
        # Generate separate word clouds for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_text = filtered_df[filtered_df['sentiment_label'] == sentiment]['comment'].str.lower()
            if not sentiment_text.empty:
                text = " ".join(sentiment_text)
                wordcloud = WordCloud(width=800, height=400, background_color="white",
                                    colormap={'positive': 'Greens', 'negative': 'Reds', 'neutral': 'Blues'}[sentiment]).generate(text)
                st.subheader(f"{sentiment.capitalize()} Comments")
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
else:
    st.warning("No comments to display for word cloud.")

# Keyword Search Results
if keyword:
    st.header("Keyword Search Results")
    if not filtered_df.empty:
        display_columns = ['comment', 'domain', 'sentiment_label', 'vader_score']
        if 'comment_id' in filtered_df.columns:
            display_columns.insert(0, 'comment_id')
        # Highlight keyword in comments
        filtered_df['comment'] = filtered_df['comment'].apply(
            lambda x: x.replace(keyword, f"**{keyword}**") if isinstance(x, str) and keyword.lower() in x.lower() else x
        )
        st.dataframe(filtered_df[display_columns], use_container_width=True)
    else:
        st.warning(f"No comments found containing '{keyword}'.")

# Sentiment Intensity Breakdown
st.header("Sentiment Intensity Breakdown")
if not filtered_df.empty and 'vader_score' in filtered_df.columns:
    avg_scores = filtered_df.groupby('sentiment_label')['vader_score'].mean().reset_index()
    st.write("Average VADER Compound Scores by Sentiment:")
    st.dataframe(avg_scores, use_container_width=True)
    # Show sample comments with scores
    display_columns = ['comment', 'domain', 'sentiment_label', 'vader_score']
    if 'comment_id' in filtered_df.columns:
        display_columns.insert(0, 'comment_id')
    st.write("Sample Comments with VADER Scores:")
    st.dataframe(filtered_df[display_columns].head(10), use_container_width=True)
else:
    st.warning("No sentiment intensity data available.")

# Actionable Recommendations
st.header("Actionable Recommendations")
if not filtered_df.empty:
    recommendations = []
    for domain in filtered_df['domain'].unique():
        domain_df = filtered_df[filtered_df['domain'] == domain]
        neg_count = len(domain_df[domain_df['sentiment_label'] == 'negative'])
        pos_count = len(domain_df[domain_df['sentiment_label'] == 'positive'])
        if neg_count > pos_count:
            recommendations.append(f"**{domain.capitalize()}**: Address negative feedback (e.g., improve reliability or clarity) to enhance user satisfaction.")
        elif pos_count > neg_count:
            recommendations.append(f"**{domain.capitalize()}**: Leverage strong positive feedback (e.g., expand successful features) to attract more users.")
        else:
            recommendations.append(f"**{domain.capitalize()}**: Balance neutral or mixed feedback by enhancing standout features and addressing minor issues.")
    for rec in recommendations:
        st.markdown(rec)
else:
    st.warning("No data available for recommendations.")

# Table: Sample Comments
st.header("Sample Comments")
if not filtered_df.empty:
    display_columns = ['comment', 'domain', 'sentiment_label', 'vader_score']
    if 'comment_id' in filtered_df.columns:
        display_columns.insert(0, 'comment_id')
    st.dataframe(
        filtered_df[display_columns].head(10),
        use_container_width=True
    )
else:
    st.warning("No comments to display for the selected filters.")

# Downloadable Insights Report
st.header("Download Insights Report")
if not filtered_df.empty:
    # Create summary for download
    summary_data = {
        'Metric': ['Total Comments', 'Positive Comments', 'Negative Comments', 'Neutral Comments'],
        'Value': [total_comments, f"{positive_comments} ({pos_percentage:.1f}%)",
                  f"{negative_comments} ({neg_percentage:.1f}%)", f"{neutral_comments} ({neu_percentage:.1f}%)"]
    }
    summary_df = pd.DataFrame(summary_data)
    csv = pd.concat([summary_df, filtered_df[display_columns]]).to_csv(index=False)
    st.download_button(
        label="Download Insights Report as CSV",
        data=csv,
        file_name="feedback_insights.csv",
        mime="text/csv"
    )
else:
    st.warning("No data available to download.")

# Interesting Fact
st.header("Interesting Fact")
if data_source == "Upload Your Dataset" and uploaded_df is not None:
    st.markdown("""
    Your uploaded dataset has been analyzed with VADER sentiment analysis, capturing positive, negative, and neutral feedback. Positive comments often highlight personalized and empathetic responses, neutral ones note functional but unremarkable experiences, and negative ones point to technical or usability issues. Explore the visualizations to uncover trends specific to your data!
    """)
else:
    st.markdown("""
    The default dataset reveals a striking contrast: the **accessibility** domain receives unanimous positive feedback (e.g., “empowering for rural patients” and “inclusive braille PDF exports”), while **technical** issues dominate negative comments (e.g., “app crashes frequently”). Neutral feedback often highlights functional but unremarkable features, suggesting opportunities to enhance user engagement.
    """)

# Conclusion
st.header("Conclusion")
if data_source == "Upload Your Dataset" and uploaded_df is not None:
    st.markdown("""
    Your dataset analysis reveals strengths, challenges, and neutral perspectives in your feedback. Use these insights to optimize services, enhance user experiences, or share with stakeholders across any industry.
    """)
else:
    st.markdown("""
    The default dataset showcases a platform excelling in accessibility and specialized care (e.g., **mental_health** and **accessibility** domains), with neutral feedback indicating functional but uninspiring features. Technical and privacy challenges (e.g., “app crashes” and “unclear privacy policies”) hinder its impact. Addressing these while amplifying strengths will boost user trust and adoption across diverse fields.
    """)
