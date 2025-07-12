import os
import glob
import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16
})

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
        return "grüne"
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
    

def merge_comments_with_topics(all_comments_df_valid, topic_dir):

    topic_files = glob.glob(os.path.join(topic_dir, "*.csv"))
    topic_dataframes = {}

    for topic_file in topic_files:
        print("READING: {topic_file}")
        partei = os.path.basename(topic_file).replace(".csv", "")
        partei = partei.lower()
        print(partei)
        try:
            df_topic = pd.read_csv(topic_file)
            df_topic["partei"] = partei
            topic_dataframes[partei] = df_topic
        except Exception as e:
            print(f"Error in reading {topic_file}: {e}")

    merged_comment_topic = []

    for partei, df_topic in topic_dataframes.items():
        comments = all_comments_df_valid[all_comments_df_valid["party"] == partei].copy()
        print(f"PARTEI: {partei}, Anzahl Kommentare: {len(comments)}")
        if comments.empty:
            print(f"No comments for {partei}")
            continue

        df_merged = pd.merge(
            comments,
            df_topic[["id", "topic_clean"]],
            left_on="video_id",
            right_on="id",
            how="left"
        )
        print(f"Number of merged comments for {partei}: {len(df_merged)}")
        print(f"Length of topic df before merge for {partei}: {len(df_topic)}")

        df_merged["topic_clean"] = df_merged["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df_merged = df_merged.explode("topic_clean")
        df_merged = df_merged.dropna(subset=["topic_clean"])
        df_merged["partei"] = partei
        merged_comment_topic.append(df_merged)
        print(f"{partei}: Vor explode – Gesamt: {len(df_merged)} | Mit Topic: {df_merged['topic_clean'].notna().sum()}")

    if not merged_comment_topic:
        print("No merged data found.")
        return None

    return pd.concat(merged_comment_topic, ignore_index=True)

def plot_heatmap(data, title, output_dir, filename):
    if data.empty:
        print(f"No data for {title}, heatmap skipped.")
        return

    # Parteinamen umbenennen
    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }
    data = data.rename(index=party_labels)

    sentiment_cmap = LinearSegmentedColormap.from_list(
        "custom_sentiment",
        ["#E3711A", "#E0E0E0", "#069770"],
        N=256
    )

    # Plot
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=sentiment_cmap,
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Stimmung"}
    )

    # Titel und Achsentexte
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Themenbereich", fontsize=16)
    ax.set_ylabel("Partei", fontsize=16)

    # X- und Y-Achsenticks beschriften
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {filename}")

def analyze_sentiment_by_topic(df_sentiment_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Final Sentiment
    sentiment_final = (
        df_sentiment_topic
        .groupby(["partei", "topic_clean"])["final_sentiment"]
        .mean()
        .unstack(fill_value=np.nan)
        .round(3)
    )
    print("Durchschnittliches Finales Sentiment pro Topic & Partei:")
    print(sentiment_final)
    plot_heatmap(sentiment_final, "Ø Kombinierte Stimmung pro Themenbereich & Partei",output_dir, "final_sentiment_by_topic_party.png")

    # Sentiment Num
    sentiment_text = (
        df_sentiment_topic
        .groupby(["partei", "topic_clean"])["sentiment_num"]
        .mean()
        .unstack(fill_value=np.nan)
        .round(3)
    )
    print("\n📊 Ø Text-Sentiment (sentiment_num) pro Topic & Partei:")
    print(sentiment_text)
    plot_heatmap(sentiment_text, "Ø Text-Sentiment pro Topic & Partei", output_dir, "text_sentiment_by_topic_party.png")

    # Emoji Sentiment
    sentiment_emoji = (
        df_sentiment_topic
        .groupby(["partei", "topic_clean"])["emoji_sentiment"]
        .mean()
        .unstack(fill_value=np.nan)
        .round(3)
    )
    print("\n📊 Ø Emoji-Sentiment pro Topic & Partei:")
    print(sentiment_emoji)
    plot_heatmap(sentiment_emoji, "Ø Emoji-Sentiment pro Topic & Partei", output_dir, "emoji_sentiment_by_topic_party.png")

def plot_emoji_sentiment_distribution(all_comments_df, start_date, end_date):
    # Farben für Sentiments
    sentiment_colors = {
        "negativ": "#F9BF93",     # Orangebraun
        "neutral": "#C1C1C1",     # Grau
        "positiv": "#069770"      # Blaugrün
    }

    # Mapping von Parteikürzeln auf vollständige Namen
    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }

    # Gruppiere nach Partei und emoji_sentiment, normalisiert (prozentual)
    emoji_sentiment_dist = (
        all_comments_df
        .groupby("party")["emoji_sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Umbenennen der Spalten: -1 = negativ, 0 = neutral, 1 = positiv
    emoji_sentiment_dist.rename(columns={
        -1.0: "negativ",
        0.0: "neutral",
        1.0: "positiv"
    }, inplace=True)

    # Spalten in gewünschter Reihenfolge
    ordered_cols = ["negativ", "neutral", "positiv"]
    emoji_sentiment_dist = emoji_sentiment_dist[ordered_cols]

    # Erzeuge die Bar-Plot-Achse
    ax = emoji_sentiment_dist.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        color=[sentiment_colors[col] for col in ordered_cols]
    )

    # Setze neue xtick-Labels mit Partei-Vollnamen
    tick_positions = range(len(emoji_sentiment_dist.index))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [party_labels.get(party, party) for party in emoji_sentiment_dist.index],
        rotation=45,
        ha="right",
        fontsize=16
    )

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # Achsen- und Diagrammtitel
    plt.title("Stimmung der Emojis pro Partei", fontsize=16, fontweight="bold", pad=10)
    plt.ylabel("Anteil", fontsize=16)
    plt.xlabel("Partei", fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title="Emoji-Stimmung", title_fontsize=16, fontsize=16, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # Speichern
    output_dir = f"plots/sentiment_analysis/{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/emoji_sentiment_verteilung.png", bbox_inches="tight")
    plt.close()

def plot_text_sentiment_distribution(all_comments_df, start_date, end_date):
    # Farben für Sentiment-Kategorien
    sentiment_colors = {
        "negativ": "#F9BF93",     # Orangebraun
        "neutral": "#C1C1C1",     # Grau
        "positiv": "#069770"      # Blaugrün
    }

    # Mapping von Parteikürzeln auf vollständige Namen
    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }

    # Gruppieren, normalisieren und Sentimentverteilung berechnen
    sentiment_dist = (
        all_comments_df
        .groupby("party")["sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    sentiment_dist.rename(columns={
        "negative": "negativ",
        "neutral": "neutral",
        "positive": "positiv"
    }, inplace=True)

    # Spaltenreihenfolge erzwingen (falls nötig)
    ordered_cols = ["negativ", "neutral", "positiv"]
    sentiment_dist = sentiment_dist.reindex(columns=ordered_cols)

    # X-Achsen-Labels: Parteivollnamen
    tick_positions = range(len(sentiment_dist.index))
    party_names = [party_labels.get(party, party) for party in sentiment_dist.index]

    # Plot erzeugen
    ax = sentiment_dist.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=[sentiment_colors[col] for col in ordered_cols]
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(party_names, rotation=45, ha="right", fontsize=16)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    plt.title("Stimmung in den Texten pro Partei", fontsize=16, fontweight="bold", pad=10)
    plt.ylabel("Anteil", fontsize=16)
    plt.xlabel("Partei", fontsize=16)
    plt.yticks(fontsize=16)

    # Legende rechts außen
    plt.legend(
        title="Stimmung",
        title_fontsize=16,
        fontsize=16,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5)
    )

    plt.tight_layout()

    # Speichern
    output_dir = f"plots/sentiment_analysis/{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/sentiment_verteilung.png", bbox_inches="tight")
    plt.close()

def plot_final_sentiment_distribution(all_comments_df_valid, start_date, end_date):
    # Farben für Sentiment-Kategorien
    sentiment_colors = {
        "negativ": "#F9BF93",     # Orangebraun
        "neutral": "#C1C1C1",     # Grau
        "positiv": "#069770"      # Blaugrün
    }

    # Mapping von Parteikürzeln auf vollständige Namen
    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }

    all_comments_df_valid["final_sentiment_label"] = all_comments_df_valid["final_sentiment"].apply(final_sentiment_label)

    # Gruppieren und normalisieren
    final_sentiment_dist = (
        all_comments_df_valid
        .groupby("party")["final_sentiment_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    ordered_cols = ["negativ", "neutral", "positiv"]
    final_sentiment_dist = final_sentiment_dist.reindex(columns=ordered_cols)

    tick_positions = range(len(final_sentiment_dist.index))
    party_names = [party_labels.get(party, party) for party in final_sentiment_dist.index]

    # Plot erzeugen
    ax = final_sentiment_dist.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=[sentiment_colors[col] for col in ordered_cols]
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(party_names, rotation=45, ha="right", fontsize=16)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    plt.title("Verteilung der kombinierten Stimmung pro Partei", fontsize=16, fontweight="bold", pad = 10)
    plt.ylabel("Anteil", fontsize=16)
    plt.xlabel("Partei", fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(
        title="Stimmung",
        title_fontsize=16,
        fontsize=16,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5)
    )

    plt.tight_layout()

    # Speichern
    output_dir = f"plots/sentiment_analysis/{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/final_sentiment_verteilung.png", bbox_inches="tight")
    plt.close()

def plot_sentiment_crosstab_heatmap(all_comments_df, start_date, end_date, output_dir=None):
    if output_dir is None:
        output_dir = f"plots/sentiment_analysis/{start_date}_{end_date}"
    os.makedirs(output_dir, exist_ok=True)

    print(all_comments_df['sentiment_num'])

    # Filter: nur valide Emoji- und Text-Sentiments
    comparison_df = all_comments_df[
        all_comments_df["emoji_sentiment"].isin([-1, 0, 1]) &
        all_comments_df["sentiment_num"].isin([-1, 0, 1])
    ]

    sentiment_counts = pd.crosstab(
        comparison_df["emoji_sentiment"],
        comparison_df["sentiment_num"],
        rownames=["Emoji-Sentiment"],
        colnames=["Text-Sentiment"]
    )

    sentiment_percent = sentiment_counts / sentiment_counts.values.sum()

    sentiment_percent = sentiment_percent.reindex(index=[-1, 0, 1], columns=[-1, 0, 1])

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        sentiment_percent,
        annot=True,
        fmt=".2%",
        cmap="PuBuGn",
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )

    sentiment_labels = {-1: "negativ", 0: "neutral", 1: "positiv"}
    x_labels = [sentiment_labels.get(x, x) for x in sentiment_percent.columns]
    y_labels = [sentiment_labels.get(y, y) for y in sentiment_percent.index]

    # Plot
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        sentiment_percent,
        annot=True,
        fmt=".2%",
        cmap="PuBuGn",
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title("Prozentuale Kreuztabelle (Emoji vs. Text-Sentiment)", fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("Text-Sentiment", fontsize=16)
    ax.set_ylabel("Emoji-Sentiment", fontsize=16)

    ax.set_xticklabels(x_labels, fontsize=16, rotation=0, ha="center")
    ax.set_yticklabels(y_labels, fontsize=16, rotation=0, va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_prozent_gesamt.png"))
    plt.close()

def main():
    start_date = config.start_date
    end_date = config.end_date
    folder = f"results/sentiment_analysis/{start_date}_{end_date}_with_emoji_sentiment"
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)

    print("Gefundene Dateien:", files)

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
    plot_emoji_sentiment_distribution(all_comments_df, start_date, end_date)

    # 2. Plot: sentiment pro Partei
    plot_text_sentiment_distribution(all_comments_df, start_date, end_date)

    # 3. Plot: final_sentiment pro Partei
    plot_final_sentiment_distribution(all_comments_df_valid, start_date, end_date)

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

    # 5. Plot: Wöchentlicher Verlauf der durchschnittlichen Stimmung pro Partei
    # Konvertiere create_time zu datetime
    all_comments_df_valid["create_time"] = pd.to_datetime(all_comments_df_valid["create_time"], errors="coerce")
    all_comments_df_valid = all_comments_df_valid.dropna(subset=["create_time"])

    # Erstelle Kalenderwoche (Startdatum der Woche)
    all_comments_df_valid["week"] = all_comments_df_valid["create_time"].dt.to_period("W").dt.start_time

    # Gruppiere: Ø final_sentiment pro Partei und Woche
    weekly_sentiment = (
        all_comments_df_valid
        .groupby(["party", "week"])["final_sentiment"]
        .mean()
        .reset_index()
    )

    # Plot für jede Partei
    parties = weekly_sentiment["party"].unique()
    plt.figure(figsize=(10, 6))
    for party in parties:
        party_data = weekly_sentiment[weekly_sentiment["party"] == party]
        plt.plot(party_data["week"], party_data["final_sentiment"], label=party)

    plt.title("Wöchentlicher Verlauf der durchschnittlichen final_sentiment pro Partei")
    plt.xlabel("Woche")
    plt.ylabel("Durchschnittliches Sentiment")
    plt.legend(title="Partei")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/sentiment_analysis/{start_date}_{end_date}/woechentlich_final_sentiment.png")
    plt.close()

    # 6. Kreuztabellen und Heatmap
    plot_sentiment_crosstab_heatmap(all_comments_df, start_date, end_date)

    # 7. Unterschied Emoji vs. Text-Sentiment pro Partei
    coherence_df = all_comments_df[
    all_comments_df["emoji_sentiment"].isin([-1, 0, 1]) &
    all_comments_df["sentiment_num"].isin([-1, 0, 1])
    ].copy()
    coherence_df["match"] = coherence_df["emoji_sentiment"] == coherence_df["sentiment_num"]

    coherence_rate = coherence_df.groupby("party")["match"].mean()
    print("\nKohärenzrate emoji vs. text pro Partei:")
    print(coherence_rate)


    # 8. Zusammenhang Engagement und Stimmung pro Video
    video_sentiment = all_comments_df_valid.groupby("video_id")["final_sentiment"].mean().reset_index()

    video_counts = all_comments_df_valid["video_id"].value_counts().reset_index()
    video_counts.columns = ["video_id", "comment_count"]

    # Zusammenführen
    video_stats = pd.merge(video_sentiment, video_counts, on="video_id")

    # Scatterplot
    plt.figure(figsize=(8, 5))
    plt.scatter(video_stats["comment_count"], video_stats["final_sentiment"])
    plt.xlabel("Anzahl Kommentare")
    plt.ylabel("Durchschnittliches Final Sentiment")
    plt.title("Engagement vs. Stimmung pro Video")
    plt.tight_layout()
    plt.savefig(f"plots/sentiment_analysis/{start_date}_{end_date}/engagement_vs_sentiment.png")
    plt.close()

   # 9. Emoji-Nutzung berechnen: Anteil der Kommentare mit mind. einem Emoji
    all_comments_df["has_emoji"] = ~all_comments_df["emoji_sentiment"].isna()
    all_comments_df["has_blue_heart"] = all_comments_df["extracted_emojis"].apply(has_blue_heart)

    # Anteil pro Partei
    emoji_comment_share = (
        all_comments_df
        .groupby("party")[["has_emoji", "has_blue_heart"]]
        .mean()
        .reset_index()
        .sort_values("party")
    )

    # Ausgabe in Prozent
    emoji_comment_share["has_emoji_percent"] = (emoji_comment_share["has_emoji"] * 100).round(2)
    emoji_comment_share["has_blue_heart_percent"] = (emoji_comment_share["has_blue_heart"] * 100).round(2)

    # 💙-Anteil unter Emoji-Kommentaren
    emoji_comment_share["share_blue_among_emojis"] = (
        emoji_comment_share["has_blue_heart"] / emoji_comment_share["has_emoji"]
    )
    emoji_comment_share["share_blue_among_emojis_percent"] = (
        emoji_comment_share["share_blue_among_emojis"] * 100
    ).round(2)

    # Ausgabe
    print("\nAnteil der Kommentare mit Emojis pro Partei:")
    print(emoji_comment_share[["party", "has_emoji_percent"]])

    print("\nAnteil der Kommentare mit 💙 pro Partei:")
    print(emoji_comment_share[["party", "has_blue_heart_percent"]])

    print("\n💙-Anteil unter Kommentaren mit Emojis pro Partei:")
    print(emoji_comment_share[["party", "share_blue_among_emojis_percent"]])

    print(f"\nGesamtzahl aller Kommentare: {len(all_comments_df)}")

    topic_dir = os.path.join("results", "topic_analysis", "merged")
    output_sentiment_topic_dir = os.path.join("plots", "sentiment_analysis", f"{start_date}_{end_date}")

    df_sentiment_topic = merge_comments_with_topics(all_comments_df_valid, topic_dir)
    if df_sentiment_topic is not None:
        analyze_sentiment_by_topic(df_sentiment_topic, output_sentiment_topic_dir)


if __name__ == "__main__":
    main()