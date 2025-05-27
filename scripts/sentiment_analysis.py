import pandas as pd
import config

from germansentiment import SentimentModel

model = SentimentModel()
batch_size = 20

for user in config.usernames:
    try:
        print(f"Lade Kommentare für {user}...")
        df_comments = pd.read_csv(f"data/data_raw/comments/{user}_comments.csv")
    except FileNotFoundError:
        print(f"Kommentar-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")
        continue

    try:
        print(f"Verarbeite Sentiment für {user}...")
        sentiments = []
        texts = df_comments['text'].astype(str).tolist()
        for i in range(0, len(texts), batch_size):
            print(f"Verarbeite Batch {i // batch_size + 1} von {len(texts) // batch_size + 1}...")
            batch = texts[i:i+batch_size]
            sentiments.extend(model.predict_sentiment(batch))

        df_comments['sentiment'] = sentiments
        df_comments.to_csv(f"data/data_processed/comments/{user}_comments_with_sentiment.csv", index=False)
    except KeyError:
        print(f"Fehler beim Hinzufügen der Sentiment-Analyse für {user}.")