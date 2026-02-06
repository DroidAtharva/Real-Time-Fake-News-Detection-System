import streamlit as st
import pandas as pd
import re
import string
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import pytz
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download('stopwords')
nltk.download('wordnet')


API_KEY = "5d804f686fac481b9d0be0aba6c883f3"


def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words).strip()


@st.cache_data
def load_data():
    try:
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")
        manual_df = pd.read_csv("manual_testing.csv")

        
        fake_df['label'] = 1
        true_df['label'] = 0

       
        fake_df = fake_df.rename(columns={'title': 'headline'})
        true_df = true_df.rename(columns={'title': 'headline'})
        manual_df = manual_df.rename(columns={'text': 'headline'})  
        manual_df['label'] = -1  

    
        combined_df = pd.concat([fake_df[['headline', 'label']], true_df[['headline', 'label']]], ignore_index=True)
        combined_df.dropna(inplace=True)
        combined_df.drop_duplicates(inplace=True)

        return combined_df
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return pd.DataFrame(columns=["headline", "label"])


@st.cache_resource
def train_models(df):
    X = df['headline'].apply(clean_text)
    y = df['label']
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  # Use bigrams for better context
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))

    return (rf_model, gb_model, vectorizer, rf_accuracy, gb_accuracy)


def detect_fake_news_combined(headline, rf_model, gb_model, vectorizer):
    cleaned = clean_text(headline)
    vect = vectorizer.transform([cleaned])
    rf_prediction = rf_model.predict(vect)[0]
    gb_prediction = gb_model.predict(vect)[0]


    prediction = "REAL" if rf_prediction + gb_prediction <= 1 else "FAKE"
    return prediction

def convert_to_ist_with_day(published_at):
    utc_time = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
    utc_time = pytz.utc.localize(utc_time)
    ist_time = utc_time.astimezone(pytz.timezone("Asia/Kolkata"))
    day = ist_time.strftime("%A") 
    time = ist_time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{day}, {time}"

def fetch_live_news(api_key, query="India Pakistan war", num_articles=5):
    try:
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
        from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")  

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&"
            f"language=en&"
            f"pageSize={num_articles}&"
            f"sortBy=publishedAt&"
            f"from={from_date}&"
            f"to={to_date}&"
            f"apiKey={api_key}"
        )
        response = requests.get(url)
        articles = response.json().get("articles", [])
        news_data = []
        for article in articles:
            title = article.get("title", "No Title Available")
            published_at = article.get("publishedAt", "N/A")
            if published_at != "N/A":
                published_time = convert_to_ist_with_day(published_at)
            else:
                published_time = "Unknown"
            news_data.append({"title": title, "published_at": published_time})
        return news_data
    except Exception as e:
        st.error(f"Error fetching live news: {e}")
        return []

def main():
    st.title("Enhanced Real-Time Fake News Detection App with Combined Models")

    df = load_data()
    if df.empty:
        st.warning("No data available to train the models.")
        return

    rf_model, gb_model, vectorizer, rf_acc, gb_acc = train_models(df)
    st.success(f" Models trained successfully! Random Forest Accuracy: {rf_acc * 100:.2f}%, Gradient Boosting Accuracy: {gb_acc * 100:.2f}%")

    choice = st.radio(
        "Choose an option:", 
        ('Type a news headline manually', 'Fetch and test live news headlines')
    )

    if choice == 'Type a news headline manually':
        headline = st.text_input("Enter the news headline:")
        if headline:
            result = detect_fake_news_combined(headline, rf_model, gb_model, vectorizer)
            st.write(f"Prediction: **{result}**")

    elif choice == 'Fetch and test live news headlines':
        with st.spinner("Fetching latest news..."):
            query = st.text_input("Enter a query for live news (default: 'India Pakistan war'):", "India Pakistan war")
            num_articles = st.slider("Number of articles to fetch:", min_value=1, max_value=20, value=5)
            news_data = fetch_live_news(API_KEY, query=query, num_articles=num_articles)
        st.write("### 🧪 Live News Predictions:")
        for news in news_data:
            result = detect_fake_news_combined(news['title'], rf_model, gb_model, vectorizer)
            st.write(f"**Headline:** {news['title']}")
            st.write(f"Published at: {news['published_at']}")
            st.write(f"Prediction: **{result}**\n")

if __name__ == "__main__":
    main()