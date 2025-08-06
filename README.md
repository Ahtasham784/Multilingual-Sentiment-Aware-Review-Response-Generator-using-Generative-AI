# Multilingual-Sentiment-Aware-Review-Response-Generator-using-Generative-AI
# STEP 1: Upload your Kaggle API token (kaggle.json)
from google.colab import files
files.upload()  # Upload kaggle.json here

# STEP 2: Setup Kaggle API credentials
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# STEP 3: Download the dataset from Kaggle
!kaggle datasets download -d mexwell/amazon-reviews-multi

# STEP 4: Unzip the downloaded dataset
!unzip -q "amazon-reviews-multi.zip" -d amazon_data

# STEP 5: Check extracted files
import os
os.listdir("amazon_data")
# Load Extracted File
import pandas as pd

train_df = pd.read_csv("amazon_data/train.csv")
test_df = pd.read_csv("amazon_data/test.csv")
val_df = pd.read_csv("amazon_data/validation.csv")

train_df.head()
# Install Libraries
!pip install transformers sentencepiece langdetect spacy nltk
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
!python -m spacy download fr_core_news_sm
# Language Detection
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
# Translation to English (if needed)
from transformers import MarianMTModel, MarianTokenizer

# Map language codes to Hugging Face models
translation_models = {
    'de': 'Helsinki-NLP/opus-mt-de-en',
    'fr': 'Helsinki-NLP/opus-mt-fr-en'
}

# Cache loaded models
loaded_models = {}

def translate_to_english(text, src_lang):
    if src_lang not in translation_models:
        return text  # Already English or unsupported
    model_name = translation_models[src_lang]
    if model_name not in loaded_models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[model_name] = (tokenizer, model)
    tokenizer, model = loaded_models[model_name]
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
# Text Cleaning (Lowercase, Stopwords, Lemmatize)
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy models
nlp_en = spacy.load("en_core_web_sm")

def clean_text(text, lang='en'):
    if lang == 'de':
        nlp = spacy.load("de_core_news_sm")
        stops = stopwords.words('german')
    elif lang == 'fr':
        nlp = spacy.load("fr_core_news_sm")
        stops = stopwords.words('french')
    else:
        nlp = nlp_en
        stops = stopwords.words('english')

    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stops]
    return ' '.join(tokens)
# Full Preprocessing Pipeline
def preprocess_pipeline(text):
    lang = detect_language(text)
    translated = translate_to_english(text, lang)
    cleaned = clean_text(translated, lang='en')  # Clean as English after translation
    return cleaned

# Apply to Dataset
# Work on just 100 samples
sample_df = train_df.sample(n=100, random_state=42).copy()

sample_df['cleaned_review'] = sample_df['review_body'].apply(preprocess_pipeline)
sample_df[['review_body', 'cleaned_review']].head()
# Topic Modeling
!pip install bertopic
!pip install sentence-transformers
!pip install umap-learn

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Use a multilingual embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize BERTopic
topic_model = BERTopic(embedding_model=embedding_model, verbose=True)

# Fit the model (this may take a while)
topics, _ = topic_model.fit_transform(sample_df['cleaned_review'])

# View topics
topic_model.get_topic_info().head()
# Sentiment Detection (Multilingual)
from transformers import pipeline

# Load sentiment pipeline (multilingual model)
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Apply sentiment prediction
sample_df['sentiment'] = sample_df['cleaned_review'].apply(lambda x: sentiment_model(x[:512])[0]['label'])

# Map to positive/negative/neutral
def map_sentiment(label):
    if '1' in label or '2' in label:
        return 'negative'
    elif '3' in label:
        return 'neutral'
    else:
        return 'positive'

sample_df['sentiment_label'] = sample_df['sentiment'].apply(map_sentiment)
# Review Clustering by Topic & Sentiment
sample_df['topic'] = topics
grouped_reviews = sample_df.groupby(['topic', 'sentiment'])

# Example: View a cluster
grouped_reviews.get_group((0, '5 stars')).head()

# Prompt Template
def generate_prompt(review_text):
    return f"""You're a customer support agent. A customer said: "{review_text}". 
Generate a polite and empathetic response."""
# Generate with FLAN-T5 or BART
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"  # or "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_response(review_text):
    prompt = generate_prompt(review_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Apply to a few samples
train_df['generated_response'] = sample_df['cleaned_review'].apply(lambda x: generate_response(x))
# Pie Chart: Sentiment Distribution
import matplotlib.pyplot as plt

# Count of sentiment labels
sentiment_counts = sample_df['sentiment_label'].value_counts()

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()
# Word Cloud per Sentiment
from wordcloud import WordCloud

# Example: Word Cloud for each sentiment
for sentiment in sample_df['sentiment_label'].unique():
    text = " ".join(sample_df[sample_df['sentiment_label'] == sentiment]['cleaned_review'])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Sentiment: {sentiment}')
    plt.show()
# Word Cloud per topic
for topic in sample_df['topic'].unique():
    text = " ".join(sample_df[sample_df['topic'] == topic]['cleaned_review'])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Topic: {topic}')
    plt.show()
# Example response generator
def generate_response(sentiment):
    if sentiment == 'positive':
        return "Thank you for your kind feedback!"
    elif sentiment == 'negative':
        return "We're sorry to hear that. We'll work on improving!"
    else:
        return "Thanks for your input — we'll consider your thoughts."

# Build summary table
summary_data = []

for topic in sample_df['topic'].unique():
    topic_reviews = sample_df[sample_df['topic'] == topic]
    
    for sentiment in topic_reviews['sentiment_label'].unique():
        group = topic_reviews[topic_reviews['sentiment_label'] == sentiment]
        if not group.empty:
            review = group['review_body'].iloc[0]  # Sample review
            response = generate_response(sentiment)
            summary_data.append({
                'Topic': topic,
                'Sentiment': sentiment,
                'Sample Review': review,
                'Auto Response': response
            })

import pandas as pd
summary_df = pd.DataFrame(summary_data)
summary_df.head()
