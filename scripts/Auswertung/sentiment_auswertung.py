import os
import glob
import pandas as pd
import config
import numpy as np

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
    # Wenn emoji_sentiment nicht bestimmbar (np.nan), nimm nur das Textsentiment
    if pd.isna(row['emoji_sentiment']):
        return row['sentiment_num']
    if (pd.isna(row['text']) or str(row['text']).strip() == "") and row['extracted_emojis']:
        return row['emoji_sentiment']
    if row['sentiment_num'] is not None:
        return (row['sentiment_num'] + row['emoji_sentiment']) / 2
    return np.nan

def compare_sentiments(df):
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    df = df.dropna(subset=["sentiment", "emoji_sentiment"]).copy()
    df["sentiment_num"] = df["sentiment"].map(mapping)
    df["match"] = df["sentiment_num"] == df["emoji_sentiment"]
    if "text" in df.columns and "extracted_emojis" in df.columns:
        df["final_sentiment"] = df.apply(compute_final_sentiment, axis=1)
    else:
        df["final_sentiment"] = np.nan
    return df

def filter_valid_sentiments(df):
    # Filtert alle Zeilen mit NaN raus
    return df[~df["final_sentiment"].isna()]

def sentiment_shares(df):
    # Gibt Anteile >0, ==0, <0 für das übergebene DataFrame zurück
    total = len(df)
    pos = (df["final_sentiment"] > 0).sum() / total if total else 0
    neu = (df["final_sentiment"] == 0).sum() / total if total else 0
    neg = (df["final_sentiment"] < 0).sum() / total if total else 0
    return pd.Series({"share_positive": pos, "share_neutral": neu, "share_negative": neg})

def has_blue_heart(emojis):
    if isinstance(emojis, list):
        return "💙" in emojis
    if isinstance(emojis, str):
        return "💙" in emojis
    return False

def main():
    start_date = config.start_date
    end_date = config.end_date
    folder = f"results/sentiment_analysis/{start_date}_{end_date}_with_emoji_sentiment"
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)

    print("Gefundene Dateien:", files)

    results = []
    all_comments = []
    sonstige_users = []

    for file in files:
        user = get_user_from_filename(file)
        party = get_party(user)
        if party == "sonstige":
            sonstige_users.append(user)
            print(f"Unbekannte Partei für User {user}, überspringe Datei: {file}")
        df = pd.read_csv(file)
        df = compare_sentiments(df)
        df["user"] = user
        df["party"] = party
        all_comments.append(df)
        valid = filter_valid_sentiments(df)
        match_rate = df["match"].mean()
        mean_final_sentiment = valid["final_sentiment"].mean() if not valid.empty else np.nan
        results.append({
            "user": user,
            "party": party,
            "match_rate": match_rate,
            "n": len(df),
            "final_sentiment": mean_final_sentiment
        })

    results_df = pd.DataFrame(results)
    print("Vergleich sentiment vs. emoji_sentiment pro User:")
    print(results_df)

    # Gruppiert nach Partei (User-Ebene)
    party_grouped = results_df.groupby("party").agg(
        mean_match_rate=("match_rate", "mean"),
        total_comments=("n", "sum"),
        mean_final_sentiment=("final_sentiment", "mean")
    ).reset_index()
    print("\nVergleich nach Partei (User-Ebene):")
    print(party_grouped)

    # Kommentar-Ebene
    all_comments_df = pd.concat(all_comments, ignore_index=True)
    all_comments_df_valid = filter_valid_sentiments(all_comments_df)

    # Sentiment-Anteile pro Partei (Kommentar-Ebene)
    shares = all_comments_df_valid.groupby("party").apply(sentiment_shares).reset_index()
    print("\nSentiment-Anteile pro Partei (Kommentar-Ebene):")
    print(shares)

    # Anteil 💙 pro Partei und User
    all_comments_df["has_blue_heart"] = all_comments_df["extracted_emojis"].apply(has_blue_heart)
    blue_heart_by_party = all_comments_df.groupby("party")["has_blue_heart"].mean().reset_index().rename(columns={"has_blue_heart": "share_blue_heart"})
    blue_heart_by_user = all_comments_df.groupby("user")["has_blue_heart"].mean().reset_index().rename(columns={"has_blue_heart": "share_blue_heart"})

    print("\nAnteil Kommentare mit 💙 pro Partei:")
    print(blue_heart_by_party)
    print("\nAnteil Kommentare mit 💙 pro User:")
    print(blue_heart_by_user)

    print("Anzahl final_sentiment == 100:", (all_comments_df["final_sentiment"] == 100).sum())

    # Optional: Speichern
    results_df.to_csv(f"results/sentiment_analysis/{start_date}_{end_date}_sentiment_emoji_comparison_users.csv", index=False)
    party_grouped.to_csv(f"results/sentiment_analysis/{start_date}_{end_date}_sentiment_emoji_comparison_parties.csv", index=False)
    shares.to_csv(f"results/sentiment_analysis/{start_date}_{end_date}_sentiment_shares_by_party.csv", index=False)
    blue_heart_by_party.to_csv(f"results/sentiment_analysis/{start_date}_{end_date}_blue_heart_by_party.csv", index=False)
    blue_heart_by_user.to_csv(f"results/sentiment_analysis/{start_date}_{end_date}_blue_heart_by_user.csv", index=False)

    # Zeige ggf. sonstige User
    if sonstige_users:
        print("\nUser, die als 'sonstige' klassifiziert wurden:")
        print(set(sonstige_users))

if __name__ == "__main__":
    main()