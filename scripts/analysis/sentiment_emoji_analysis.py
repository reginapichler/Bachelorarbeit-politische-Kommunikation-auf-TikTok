import glob
import os
import pandas as pd
from collections import Counter
import emoji
import config_analysis as config

def count_emojis_in_file(filepath):
    df = pd.read_csv(filepath, usecols=["text"])
    counter = Counter()
    for text in df["text"].dropna():
        emojis = [e['emoji'] for e in emoji.emoji_list(str(text))]
        counter.update(emojis)
    return counter

def get_user_from_filename(filename):
    return filename.split("_")[0]

def analyze_emojis(input_dir, parteien):
    emoji_counter = Counter()
    partei_counters = {p: Counter() for p in parteien}

    for fname in os.listdir(input_dir):
        if fname.endswith(".csv"):
            user = get_user_from_filename(fname)
            filepath = os.path.join(input_dir, fname)
            file_counter = count_emojis_in_file(filepath)
            emoji_counter.update(file_counter)
            for partei, userlist in parteien.items():
                if user in userlist:
                    partei_counters[partei].update(file_counter)
    return emoji_counter, partei_counters

def save_emoji_counts(counter, path):
    df = pd.DataFrame(counter.items(), columns=["emoji", "count"]).sort_values("count", ascending=False)
    df.to_csv(path, index=False)

def preprocess_emoji_data(input_path):
    """Preprocess the emoji data und fügt eine Sentiment-Spalte hinzu."""
    emoji_data = pd.read_csv(input_path, sep=";")
    emoji_data['N'] = emoji_data['Negative'] + emoji_data['Neutral'] + emoji_data['Positive']
    emoji_data['score_pos'] = emoji_data['Positive'] / emoji_data['N']
    emoji_data['score_neg'] = emoji_data['Negative'] / emoji_data['N']
    emoji_data['score_neu'] = emoji_data['Neutral'] / emoji_data['N']

    # sentiment_label: 1 = pos, 0 = neu, -1 = neg
    emoji_data['sentiment_label'] = emoji_data[['score_neg', 'score_neu', 'score_pos']].idxmax(axis=1)
    emoji_data['sentiment_label'] = emoji_data['sentiment_label'].map({
        'score_neg': -1,
        'score_neu': 0,
        'score_pos': 1
    })

    return emoji_data

def compute_emoji_sentiment(folder_sentiments, emoji_df, folder_output):
    """
    Für jede Datei in folder_sentiments:
    - Extrahiere Emojis aus der Spalte 'text'
    - Ermittle den Sentiment-Label für jedes Emoji (Merge mit emoji_df)
    - Berechne den Mittelwert pro Kommentar (Text)
    - Schreibe das Ergebnis als neue Spalte 'emoji_sentiment' in die Datei, Fehlercode 111 wenn kein Emoji-Sentiment gefunden wird.
    """
    os.makedirs(folder_output, exist_ok=True)
    sentiment_files = glob.glob(folder_sentiments)
    for file in sentiment_files:
        df = pd.read_csv(file)
        # Emojis extrahieren
        df['emojis'] = df['text'].apply(lambda x: [e['emoji'] for e in emoji.emoji_list(str(x))])
        # Explodiere Emojis für Merge
        df_exploded = df.explode('emojis')
        # Merge mit Sentiment-Label
        merged = df_exploded.merge(emoji_df[['Emoji', 'sentiment_label']], left_on='emojis', right_on='Emoji', how='left')
        # sentiment_label auffüllen mit 111, falls kein Emoji-Sentiment gefunden
        merged['sentiment_label'] = merged['sentiment_label'].fillna(111)
        # Mittelwert pro Original-Kommentar berechnen
        emoji_sentiment = merged.groupby(merged.index)['sentiment_label'].mean()
        # Wenn keine Emojis, dann 0
        emoji_sentiment = emoji_sentiment.fillna(0)
        df['emoji_sentiment'] = emoji_sentiment

        # Neuen Dateinamen im Output-Folder mit _final.csv
        base = os.path.basename(file)
        outname = base.replace('_complete.csv', '_final.csv')
        output_path = os.path.join(folder_output, outname)

        df.to_csv(output_path, index=False)
        print(f"Emoji-Sentiment für {file} gespeichert unter: {output_path}")


if __name__ == "__main__":
    start_date = config.start_date
    end_date = config.end_date
    input_dir = "data/data_preprocessed/comments"
    input_path_emoji = "data/data_raw/emoji_sentiment_data.csv"
    input_folder_sentiment = f"results/sentiment_analysis/{start_date}_{end_date}/*.csv"
    output_folder_sentiment = f"results/sentiment_analysis/{start_date}_{end_date}_with_emoji_sentiment"
    output_dir = f"results/emoji_analysis/{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "emoji_counts.csv")

    parteien = {
        "afd": config.afd_usernames,
        "spd": config.spd_usernames,
        "cducsu": config.cdu_csu_usernames,
        "gruene": config.gruene_usernames,
        "linke": config.linke_usernames
    }

    emoji_counter, partei_counters = analyze_emojis(input_dir, parteien)

    # Gesamtdaten speichern
    save_emoji_counts(emoji_counter, output_path)
    print(f"Emoji-Liste (gesamt) gespeichert unter: {output_path}")

    # Pro Partei speichern
    for partei, counter in partei_counters.items():
        partei_path = os.path.join(output_dir, f"emoji_counts_{partei}.csv")
        save_emoji_counts(counter, partei_path)
        print(f"Emoji-Liste ({partei}) gespeichert unter: {partei_path}")

    emoji_df = preprocess_emoji_data(input_path_emoji)
    compute_emoji_sentiment(input_folder_sentiment, emoji_df, output_folder_sentiment)