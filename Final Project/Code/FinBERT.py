import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# File paths
input_file = '/Users/lihanyu/Desktop/NLP-Final/Dataset/Apple_data_value10_with_sum.csv'
output_file = '/Users/lihanyu/Desktop/NLP-Final/Dataset/Apple_sentiment_analysis.csv'

# Load data
print("Loading data...")
df = pd.read_csv(input_file)
print(f"Data loaded successfully. Total records: {len(df)}")

# Load FinBERT model
print("Loading FinBERT model...")
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
print("Model loaded successfully")

# Sentiment analysis function with truncation
def get_sentiment_score(text):
    try:
        inputs = tokenizer(text, max_length=512, truncation=True, padding=False)
        truncated_text = tokenizer.decode(inputs['input_ids'], skip_special_tokens=True)
        result = nlp(truncated_text)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error processing text: {text[:50]}... | Error: {str(e)}")
        return "neutral", 0  # Default to neutral if there's an error

# Perform sentiment analysis
print("Starting sentiment analysis...")
tqdm.pandas(desc="Progress")
df[['sentiment_label', 'sentiment_score']] = df['body'].progress_apply(
    lambda x: pd.Series(get_sentiment_score(x))
)
print("Sentiment analysis completed")

# Save results to CSV
df.to_csv(output_file, index=False)
print(f"Sentiment analysis results saved to {output_file}")
