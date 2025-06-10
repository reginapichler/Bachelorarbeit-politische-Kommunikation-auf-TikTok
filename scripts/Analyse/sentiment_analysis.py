import pandas as pd
import config_analysis as config
import os
from germansentiment import SentimentModel

def load_comments(input_path):
    try:
        print(f"Lade Kommentare für {input_path}...")
        return pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Kommentar-Daten nicht gefunden: {input_path}. Verarbeitung übersprungen.")
        return None

def compute_sentiments(texts, model, batch_size=100):
    sentiments = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        print(f"Verarbeite Batch {batch_num} von {num_batches}...")
        batch = texts[i:i+batch_size]
        batch_sentiments = model.predict_sentiment(batch)
        sentiments.extend(batch_sentiments)
    return sentiments

def process_user(user, start_date, end_date, model, batch_size):
    input_path = f"data/data_preprocessed/comments/{user}_comments_{start_date}_{end_date}_preprocessed.csv"
    output_dir = f"results/sentiment_analysis/{start_date}_{end_date}.csv"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{user}_comments_with_sentiment.csv"
    output_complete = f"{output_dir}/{user}_comments_with_sentiment_complete.csv"

    if os.path.exists(output_complete):
        print(f"Sentiment-Analyse für {user} bereits komplett vorhanden. Überspringe.")
        return

    df_comments = load_comments(input_path)
    if df_comments is None:
        return

    try:
        print(f"Verarbeite Sentiment für {user}...")
        texts = df_comments['text'].astype(str).tolist()
        sentiments = compute_sentiments(texts, model, batch_size)
        df_comments['sentiment'] = sentiments
        df_comments.to_csv(output_path, index=False)
        os.rename(output_path, output_complete)
        print(f"Sentiment-Analyse für {user} abgeschlossen und als _complete gespeichert.")
    except KeyError:
        print(f"Fehler beim Hinzufügen der Sentiment-Analyse für {user}.")
    except Exception as e:
        print(f"Fehler bei {user}: {e}")

def main():
    model = SentimentModel()
    batch_size = 100
    start_date = config.start_date
    end_date = config.end_date

    for user in config.usernames:
        process_user(user, start_date, end_date, model, batch_size)

if __name__ == "__main__":
    main()