import pandas as pd
import config_analysis as config
import os
from germansentiment import SentimentModel

def load_comments(input_path):
    """Load comments from a CSV file."""
    try:
        print(f"Loading comments for {input_path}...")
        return pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Comment data not found: {input_path}. Skipped.")
        return None

def compute_sentiments(texts, model, batch_size=100):
    """Compute sentiments for a list of texts using a sentiment model."""
    sentiments = []
    # process texts in batches
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num} of {num_batches} batches...")
        batch = texts[i:i+batch_size]
        batch_sentiments = model.predict_sentiment(batch)
        sentiments.extend(batch_sentiments)
    return sentiments

def process_user(user, start_date, end_date, model, batch_size):
    """Process sentiment analysis for a user."""
    input_path = f"data/data_preprocessed/comments/{user}_comments_{start_date}_{end_date}_preprocessed.csv"
    output_dir = f"results/sentiment_analysis/{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{user}_comments_with_sentiment.csv"
    output_complete = f"{output_dir}/{user}_comments_with_sentiment_complete.csv"

    if os.path.exists(output_complete):
        print(f"Sentiment analysis for {user} already done. Skipping.")
        return

    df_comments = load_comments(input_path)
    if df_comments is None:
        return

    try:
        print(f"Processing sentiment for {user}...")
        texts = df_comments['text'].astype(str).tolist()
        sentiments = compute_sentiments(texts, model, batch_size)
        df_comments['sentiment'] = sentiments
        df_comments.to_csv(output_path, index=False)
        os.rename(output_path, output_complete)
        print(f"Sentiment analysis completed for {user} and saved with prefix _complete.")
    except KeyError:
        print(f"Error when adding sentiment for {user}.")
    except Exception as e:
        print(f"Error for {user}: {e}")

def main():
    model = SentimentModel() # use german sentiment bert model of oliverguhr
    batch_size = 100
    start_date = config.start_date
    end_date = config.end_date

    for user in config.usernames:
        process_user(user, start_date, end_date, model, batch_size)

if __name__ == "__main__":
    main()