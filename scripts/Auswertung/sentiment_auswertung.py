import os
import glob
import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_user_from_filename(filename):
    base = os.path.basename(filename)
    return base.split("_comments")[0]

def get_party(user):
    if user in config.afd_usernames:
        return "afd"
    elif user in config.spd_usernames:
        return "spd"
    elif user in config.cdu_csu_usernames:
        return "cdu_csu"
    elif user in config.gruene_usernames:
        return "gruene"
    elif user in config.linke_usernames:
        return "linke"
    else:
        return "sonstige"

def compute_final_sentiment(row):
    if pd.isna(row['emoji_sentiment']):
        return row['sentiment_num']
    if (pd.isna(row['text']) or str(row['text']).strip() == "") and row['extracted_emojis']:
        return row['emoji_sentiment']
    if row['sentiment_num'] is not None:
        return (row['sentiment_num'] + row['emoji_sentiment']) / 2
    return np.nan

def has_blue_heart(emojis):
    if isinstance(emojis, list):
        return "💙" in emojis
    if isinstance(emojis, str):
        return "💙" in emojis
    return False

def final_sentiment_label(val):
    if val > 0:
        return "positiv"
    elif val < 0:
        return "negativ"
    else:
        return "neutral"

def emoji_sentiment_conversion(val):
    if pd.isna(val):
        return np.nan
    if val < -0.2:
        return -1
    elif val > 0.2:
        return 1
    else:
        return 0

def main():
    start_date = config.start_date
    end_date = config.end_date
    folder = f"results/sentiment_analysis/{start_date}_{end_date}_with_emoji_sentiment"
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)

    print("Gefundene Dateien:", files)

    # --- Einlesen und Vereinheitlichung ---
    all_comments = []
    sonstige_users = set()

    for file in files:
        user = get_user_from_filename(file)
        party = get_party(user)
        if party == "sonstige":
            sonstige_users.add(user)
            print(f"Unbekannte Partei für User {user}, überspringe Datei: {file}")
            continue
        df = pd.read_csv(file)
        # Sentiment-Mapping und finale Berechnung
        df["sentiment_num"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
        df["emoji_sentiment"] = df["emoji_sentiment"].map(lambda x: emoji_sentiment_conversion(x) if not pd.isna(x) else np.nan)
        df["match"] = df["sentiment_num"] == df["emoji_sentiment"]
        df["final_sentiment"] = df.apply(compute_final_sentiment, axis=1)
        df["user"] = user
        df["party"] = party
        df["has_blue_heart"] = df["extracted_emojis"].apply(has_blue_heart)
        all_comments.append(df)

    if not all_comments:
        print("Keine gültigen Kommentare gefunden.")
        return

    all_comments_df = pd.concat(all_comments, ignore_index=True)
    all_comments_df_valid = all_comments_df[~all_comments_df["final_sentiment"].isna()]

    # --- Plots und Auswertungen ---

    # 1. Plot: emoji_sentiment pro Partei
    emoji_sentiment_dist = all_comments_df.groupby("party")["emoji_sentiment"].value_counts(normalize=True).unstack(fill_value=0)
    print("\nVerteilung emoji_sentiment pro Partei:")
    print(emoji_sentiment_dist)
    emoji_sentiment_dist.plot(kind="bar", stacked=True, figsize=(8,5), color=["#d62728", "#ff7f0e", "#2ca02c"])
    plt.title("Verteilung emoji_sentiment pro Partei")
    plt.ylabel("Anteil")
    plt.xlabel("Partei")
    plt.legend(title="emoji_sentiment")
    plt.tight_layout()
    os.makedirs(f"plots/sentiment_analysis", exist_ok=True)
    plt.savefig(f"plots/sentiment_analysis/{start_date}_{end_date}/emoji_sentiment_verteilung.png")
    plt.close()

    # 2. Plot: sentiment pro Partei
    sentiment_dist = all_comments_df.groupby("party")["sentiment"].value_counts(normalize=True).unstack(fill_value=0)
    print("\nVerteilung sentiment pro Partei:")
    print(sentiment_dist)
    sentiment_dist.plot(kind="bar", stacked=True, figsize=(8,5), color=["#d62728", "#ff7f0e", "#2ca02c"])
    plt.title("Verteilung sentiment pro Partei")
    plt.ylabel("Anteil")
    plt.xlabel("Partei")
    plt.legend(title="sentiment")
    plt.tight_layout()
    plt.savefig(f"plots/sentiment_analysis/{start_date}_{end_date}/sentiment_verteilung.png")
    plt.close()

    # 3. Plot: final_sentiment pro Partei
    all_comments_df_valid["final_sentiment_label"] = all_comments_df_valid["final_sentiment"].apply(final_sentiment_label)
    final_sentiment_dist = all_comments_df_valid.groupby("party")["final_sentiment_label"].value_counts(normalize=True).unstack(fill_value=0)
    print("\nVerteilung final_sentiment pro Partei:")
    print(final_sentiment_dist)
    final_sentiment_dist.plot(kind="bar", stacked=True, figsize=(8,5), color=["#d62728", "#ff7f0e", "#2ca02c"])
    plt.title("Verteilung final_sentiment pro Partei")
    plt.ylabel("Anteil")
    plt.xlabel("Partei")
    plt.legend(title="final_sentiment")
    plt.tight_layout()
    plt.savefig(f"plots/sentiment_analysis/{start_date}_{end_date}/final_sentiment_verteilung.png")
    plt.close()

    # 4. Plot: Anteil 💙 pro Partei (Kommentar-Ebene)
    blue_heart_by_party = all_comments_df.groupby("party")["has_blue_heart"].mean().reset_index()
    print("\nAnteil 💙 pro Partei (Kommentar-Ebene):")
    print(blue_heart_by_party)
    blue_heart_by_party.set_index("party")["has_blue_heart"].plot(kind="bar", color="#1f77b4", figsize=(8,5))
    plt.title("Anteil 💙 pro Partei (Kommentar-Ebene)")
    plt.ylabel("Anteil")
    plt.xlabel("Partei")
    plt.tight_layout()
    plt.savefig(f"plots/sentiment_analysis/{start_date}_{end_date}/blue_heart_verteilung.png")
    plt.close()

if __name__ == "__main__":
    main()