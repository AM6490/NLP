import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from tqdm import tqdm

# File paths
input_file = '/Users/lihanyu/Desktop/NLP-Final/Dataset/Apple_data_value10_with_sum.csv'
output_file = '/Users/lihanyu/Desktop/NLP-Final/Dataset/Apple_sentiment_analysis1.csv'

# Step 1: Load data
print("Loading data...")
df = pd.read_csv(input_file)
print(f"Data loaded successfully. Total records: {len(df)}")

# Step 2: Pre-train using Masked Language Modeling (MLM)
print("Starting MLM fine-tuning...")

# Load tokenizer and MLM model
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)

# Prepare data for MLM
texts = df['body'].astype(str).tolist()
dataset = Dataset.from_dict({"text": texts})

# Tokenization function with padding and truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)  # 限制 max_length

# Tokenize the data
tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define DataCollator for MLM to handle dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # 15% 的 token 被掩盖
)

# Define MLM training arguments
mlm_training_args = TrainingArguments(
    output_dir="./finbert-mlm-finetuned",
    evaluation_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=4,  
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    logging_dir="./mlm_logs",
    dataloader_num_workers=0,  
)

# Trainer for MLM fine-tuning (CPU-only mode)
trainer = Trainer(
    model=mlm_model,
    args=mlm_training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator,
)

# Start MLM fine-tuning
trainer.train()
print("MLM fine-tuning completed")

# Step 3: Sentiment Analysis using FinBERT
print("Starting sentiment analysis...")

# Load the MLM fine-tuned model for sequence classification
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "./finbert-mlm-finetuned", num_labels=3
)
nlp = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=tokenizer, device=-1)  # CPU 模式

# Sentiment analysis function
def get_sentiment_score(text):
    try:
        result = nlp(text)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Error processing text: {text[:50]}... | Error: {str(e)}")
        return "neutral", 0  # 默认返回 neutral 和 0

# Apply sentiment analysis
tqdm.pandas(desc="Progress")
df[['sentiment_label', 'sentiment_score']] = df['body'].progress_apply(
    lambda x: pd.Series(get_sentiment_score(str(x)))
)

print("Sentiment analysis completed")

# Step 4: Save results
df.to_csv(output_file, index=False)
print(f"Sentiment analysis results saved to {output_file}")
