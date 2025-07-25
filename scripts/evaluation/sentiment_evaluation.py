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
    """Extracts the username from a filename"""
    base = os.path.basename(filename)
    return base.split("_comments")[0]

def get_party(user):
    """Returns the party based on the username."""
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
    """Computes the final sentiment based on text and emoji sentiment."""
    if pd.isna(row['emoji_sentiment']):
        return row['sentiment_num']
    if (pd.isna(row['text']) or str(row['text']).strip() == "") and row['extracted_emojis']:
        return row['emoji_sentiment']
    if row['sentiment_num'] is not None:
        return (row['sentiment_num'] + row['emoji_sentiment']) / 2
    return np.nan

def has_heart(emojis, heart):
    """Checks if the heart emoji is present in the emojis."""
    if isinstance(emojis, list):
        return heart in emojis
    if isinstance(emojis, str):
        return heart in emojis
    return False

def final_sentiment_label(val):
    """Labels the final sentiment based on its value."""
    if val > 0.2:
        return "positiv"
    elif val < -0.2:
        return "negativ"
    else:
        return "neutral"

def emoji_sentiment_conversion(val):
    """Converts emoji sentiment values to labels."""
    if pd.isna(val):
        return np.nan
    if val < -0.2:
        return -1
    elif val > 0.2:
        return 1
    else:
        return 0
    

def merge_comments_with_topics(all_comments_df_valid, topic_dir):
    """Merges comments with their topics."""

    # find all files with topics
    topic_files = glob.glob(os.path.join(topic_dir, "*.csv"))
    topic_dataframes = {}

    # read the files, normalize party names, convert to dfs
    for topic_file in topic_files:
        print("READING: {topic_file}")
        party = os.path.basename(topic_file).replace(".csv", "")
        party = party.lower()
        print(party)
        try:
            df_topic = pd.read_csv(topic_file)
            df_topic["partei"] = party
            topic_dataframes[party] = df_topic
        except Exception as e:
            print(f"Error in reading {topic_file}: {e}")

    merged_comment_topic = []

    # merge comment dfs with topic dfs
    for party, df_topic in topic_dataframes.items():
        comments = all_comments_df_valid[all_comments_df_valid["party"] == party].copy()
        print(f"Party: {party}, Number of comments: {len(comments)}")
        if comments.empty:
            print(f"No comments for {party}")
            continue

        df_merged = pd.merge(
            comments,
            df_topic[["id", "topic_clean"]],
            left_on="video_id",
            right_on="id",
            how="left"
        )
        print(f"Number of merged comments for {party}: {len(df_merged)}")
        print(f"Length of topic df before merge for {party}: {len(df_topic)}")

        df_merged["topic_clean"] = df_merged["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        # create one row per topic (video data with several topics is exploded)
        df_merged = df_merged.explode("topic_clean")
        df_merged = df_merged.dropna(subset=["topic_clean"])
        df_merged["partei"] = party
        merged_comment_topic.append(df_merged)
        print(f"{party}: Length before exploding: {len(df_merged)}; With topic after exploding: {df_merged['topic_clean'].notna().sum()}")

    if not merged_comment_topic:
        print("No merged data found.")
        return None

    return pd.concat(merged_comment_topic, ignore_index=True)

def plot_heatmap(data, title, output_dir, filename):
    """Plots a heatmap between sentiment and topics for each party."""
    if data.empty:
        print(f"No data for {title}, heatmap skipped.")
        return
    
    # labels for parties
    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }
    data = data.rename(index=party_labels)

    # create own colormap
    sentiment_cmap = LinearSegmentedColormap.from_list(
        "custom_sentiment",
        ["#E3711A", "#E0E0E0", "#007038"],
        N=256
    )

    # create heatmap
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

    # set labels, titles, ...
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Themenbereich", fontsize=16)
    ax.set_ylabel("Partei", fontsize=16)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Heatmap saved: {filename}")

def analyze_sentiment_by_topic(df_sentiment_topic, output_dir):
    """Analyzes sentiment by topic and party."""
    os.makedirs(output_dir, exist_ok=True)

    # Get final sentiment by topic and party
    sentiment_final = (
        df_sentiment_topic
        .groupby(["partei", "topic_clean"])["final_sentiment"]
        .mean()
        .unstack(fill_value=np.nan)
        .round(3)
    )
    print("Mean final sentiment by topic and party:")
    print(sentiment_final)
    # create heatmap for final sentiment
    plot_heatmap(sentiment_final, "Ø Kombinierte Stimmung pro Themenbereich & Partei",output_dir, "final_sentiment_by_topic_party.png")

    # get text sentiment by topic and party
    sentiment_text = (
        df_sentiment_topic
        .groupby(["partei", "topic_clean"])["sentiment_num"]
        .mean()
        .unstack(fill_value=np.nan)
        .round(3)
    )

    print("Text sentiment by topic and party:")
    print(sentiment_text)
    # create heatmap for text sentiment
    plot_heatmap(sentiment_text, "Ø Text-Sentiment pro Topic & Partei", output_dir, "text_sentiment_by_topic_party.png")

    # get emoji sentiment by topic and party
    sentiment_emoji = (
        df_sentiment_topic
        .groupby(["partei", "topic_clean"])["emoji_sentiment"]
        .mean()
        .unstack(fill_value=np.nan)
        .round(3)
    )
    print("Emoji sentiment by topic and party:")
    print(sentiment_emoji)
    # create heatmap for emoji sentiment
    plot_heatmap(sentiment_emoji, "Ø Emoji-Sentiment pro Topic & Partei", output_dir, "emoji_sentiment_by_topic_party.png")

def plot_emoji_sentiment_distribution(all_comments_df):
    """Plots emoji sentiment distribution in stacked bar plot."""
    sentiment_colors = {
        "negativ": "#FF8800",     # Orangebraun
        "neutral": "#C1C1C1",     # Grau
        "positiv": "#007038"      # Blaugrün
    }

    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }

    # get distribution of emoji sentiment by party
    emoji_sentiment_dist = (
        all_comments_df
        .groupby("party")["emoji_sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # Rename values to labels
    emoji_sentiment_dist.rename(columns={
        -1.0: "negativ",
        0.0: "neutral",
        1.0: "positiv"
    }, inplace=True)

    # order columns
    ordered_cols = ["negativ", "neutral", "positiv"]
    emoji_sentiment_dist = emoji_sentiment_dist[ordered_cols]

    # create bar plot, stacked by sentiment
    ax = emoji_sentiment_dist.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=[sentiment_colors[col] for col in ordered_cols]
    )

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

    # set title, labels, legend
    plt.title("Stimmung der Emojis pro Partei", fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("Anteil", fontsize=16)
    plt.xlabel("Partei", fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(
        title="Emoji-Stimmung", 
        title_fontsize=16, fontsize=16, 
        loc='center left', 
        bbox_to_anchor=(1, 0.5)) # position: outside the plot
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_dir = f"plots/sentiment_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/emoji_sentiment_verteilung.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_text_sentiment_distribution(all_comments_df):
    """Plots text sentiment distribution in stacked bar plot."""
    sentiment_colors = {
        "negativ": "#FF8800",     # Orangebraun
        "neutral": "#C1C1C1",     # Grau
        "positiv": "#007038"      # Blaugrün
    }

    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }

    # get text sentiment distribution by party
    sentiment_dist = (
        all_comments_df
        .groupby("party")["sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    # rename values to german labels
    sentiment_dist.rename(columns={
        "negative": "negativ",
        "neutral": "neutral",
        "positive": "positiv"
    }, inplace=True)

    # order columns
    ordered_cols = ["negativ", "neutral", "positiv"]
    sentiment_dist = sentiment_dist.reindex(columns=ordered_cols)

    # get party names for the ticks
    tick_positions = range(len(sentiment_dist.index))
    party_names = [party_labels.get(party, party) for party in sentiment_dist.index]

    # create plot
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

    # set title, labels, legend
    plt.title("Stimmung in den Texten pro Partei", fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("Anteil", fontsize=16)
    plt.xlabel("Partei", fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(
        title="Stimmung",
        title_fontsize=16,
        fontsize=16,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5) # position: outside the plot
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7) # set grid
    plt.tight_layout()

    output_dir = f"plots/sentiment_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/sentiment_verteilung.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_final_sentiment_distribution(all_comments_df_valid):
    """Plots the final sentiment distribution in a stacked bar plot."""
    sentiment_colors = {
        "negativ": "#FF8800",     # Orangebraun
        "neutral": "#C1C1C1",     # Grau
        "positiv": "#007038"      # Blaugrün
    }

    party_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "grüne": "Die Grünen",
        "linke": "Die Linke",
        "spd": "SPD"
    }

    # get the final sentiment label with tolerance of 0.2
    all_comments_df_valid["final_sentiment_label"] = all_comments_df_valid["final_sentiment"].apply(final_sentiment_label)

    # get distribution of final sentiment by party
    final_sentiment_dist = (
        all_comments_df_valid
        .groupby("party")["final_sentiment_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    print("Final sentiment distribution:")
    print(final_sentiment_dist)

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

    plt.title("Verteilung der kombinierten Stimmung pro Partei", fontsize=16, fontweight="bold", pad = 15)
    plt.ylabel("Anteil", fontsize=16)
    plt.xlabel("Partei", fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)


    plt.legend(
        title="Stimmung",
        title_fontsize=16,
        fontsize=16,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5) # legend position: outside the plot
    )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    output_dir = f"plots/sentiment_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/final_sentiment_verteilung.png", bbox_inches="tight", dpi=300)
    plt.close()

def plot_sentiment_crosstab_heatmap(all_comments_df, output_dir="plots/sentiment_analysis"):
    """Plots a heatmap of the sentiment crosstab between emoji and text sentiment."""

    os.makedirs(output_dir, exist_ok=True)

    print(all_comments_df['sentiment_num'])

    # only use valid sentiment values
    comparison_df = all_comments_df[
        all_comments_df["emoji_sentiment"].isin([-1, 0, 1]) &
        all_comments_df["sentiment_num"].isin([-1, 0, 1])
    ]

    # count sentiment combinations
    sentiment_counts = pd.crosstab(
        comparison_df["emoji_sentiment"],
        comparison_df["sentiment_num"],
        rownames=["Emoji-Sentiment"],
        colnames=["Text-Sentiment"]
    )

    # convert to percentage
    sentiment_percent = sentiment_counts / sentiment_counts.values.sum()

    sentiment_percent = sentiment_percent.reindex(index=[-1, 0, 1], columns=[-1, 0, 1])

    # map label names
    sentiment_labels = {-1: "negativ", 0: "neutral", 1: "positiv"}
    x_labels = [sentiment_labels.get(x, x) for x in sentiment_percent.columns]
    y_labels = [sentiment_labels.get(y, y) for y in sentiment_percent.index]

    # plot heatmap
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

    # set title, labels, ...
    ax.set_title("Prozentuale Kreuztabelle (Emoji vs. Text-Sentiment)", fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("Text-Sentiment", fontsize=16)
    ax.set_ylabel("Emoji-Sentiment", fontsize=16)

    ax.set_xticklabels(x_labels, fontsize=16, rotation=0, ha="center")
    ax.set_yticklabels(y_labels, fontsize=16, rotation=0, va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_prozent_gesamt.png"), dpi=300)
    plt.close()

def plot_blue_heart_share_by_party(all_comments_df, output_dir="plots/sentiment_analysis"):
    """Plots and prints the share of comments with blue heart emoji by party."""
    blue_heart_by_party = all_comments_df.groupby("party")["has_blue_heart"].mean().reset_index()
    print("Share of blue hearts by party:")
    print(blue_heart_by_party)
    plt.figure(figsize=(8,5))
    blue_heart_by_party.set_index("party")["has_blue_heart"].plot(kind="bar", color="#1f77b4")
    # create plot
    plt.title("Anteil blaue Herzen pro Partei (Kommentar-Ebene)", fontweight="bold", pad=15)
    plt.ylabel("Anteil")
    plt.xlabel("Partei")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "blue_heart_verteilung.png"), dpi=300)
    plt.close()

def plot_weekly_final_sentiment(all_comments_df_valid, output_dir="plots/sentiment_analysis"):
    """Plots the weekly average final_sentiment per party"""
    # Convert create_time to datetime and drop missing
    all_comments_df_valid = all_comments_df_valid.copy()
    all_comments_df_valid["create_time"] = pd.to_datetime(all_comments_df_valid["create_time"], errors="coerce")
    all_comments_df_valid = all_comments_df_valid.dropna(subset=["create_time"])

    # Create week column (start date of week)
    all_comments_df_valid["week"] = all_comments_df_valid["create_time"].dt.to_period("W").dt.start_time

    # Group: average final_sentiment per party and week
    weekly_sentiment = (
        all_comments_df_valid
        .groupby(["party", "week"])["final_sentiment"]
        .mean()
        .reset_index()
    )

    # Plot for each party
    parties = weekly_sentiment["party"].unique()
    plt.figure(figsize=(8, 5))
    for party in parties:
        party_data = weekly_sentiment[weekly_sentiment["party"] == party]
        plt.plot(party_data["week"], party_data["final_sentiment"], label=party)

    plt.title("Wöchentlicher Verlauf der durchschnittlichen final_sentiment pro Partei", fontweight="bold", pad=15)
    plt.xlabel("Woche")
    plt.ylabel("Durchschnittliches Sentiment")
    plt.legend(title="Partei")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "woechentlich_final_sentiment.png"), dpi=300)
    plt.close()

def plot_engagement_vs_sentiment(all_comments_df_valid, output_dir="plots/sentiment_analysis"):
    """Plots the relationship between engagement (number of comments) and average final sentiment per video."""
    # get mean final sentiment by video and comment count
    video_sentiment = all_comments_df_valid.groupby("video_id")["final_sentiment"].mean().reset_index()
    video_counts = all_comments_df_valid["video_id"].value_counts().reset_index()
    video_counts.columns = ["video_id", "comment_count"]

    video_stats = pd.merge(video_sentiment, video_counts, on="video_id")

    # Create scatterplot 
    plt.figure(figsize=(8, 5))
    plt.scatter(video_stats["comment_count"], video_stats["final_sentiment"])
    plt.xlabel("Anzahl Kommentare")
    plt.ylabel("Durchschnittliches Final Sentiment")
    plt.title("Engagement vs. Stimmung pro Video", fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"plots/sentiment_analysis/engagement_vs_sentiment.png", dpi=300)
    plt.close()

def show_emoji_usage(all_comments_df):
    """Shows the share of comments with specific heart emojis by party."""
    all_comments_df["has_emoji"] = ~all_comments_df["emoji_sentiment"].isna()
    all_comments_df["has_blue_heart"] = all_comments_df["extracted_emojis"].apply(lambda x: has_heart(x, "💙"))
    all_comments_df["has_green_heart"] = all_comments_df["extracted_emojis"].apply(lambda x: has_heart(x, "💚"))
    all_comments_df["has_red_heart"] = all_comments_df["extracted_emojis"].apply(lambda x: has_heart(x, "❤️"))
    all_comments_df["has_pink_heart"] = all_comments_df["extracted_emojis"].apply(lambda x: has_heart(x, "🩷"))

    # share by party
    heart_cols = ["has_blue_heart", "has_green_heart", "has_red_heart", "has_pink_heart"]
    emoji_comment_share = (
        all_comments_df
        .groupby("party")[heart_cols]
        .mean()
        .reset_index()
        .sort_values("party")
    )

    for col in heart_cols:
        emoji_comment_share[f"{col}_percent"] = (emoji_comment_share[col] * 100).round(2)

    # print results
    print("\nShare of comments with blue heart per party:")
    print(emoji_comment_share[["party", "has_blue_heart_percent"]])
    print("\nShare of comments with green heart per party:")
    print(emoji_comment_share[["party", "has_green_heart_percent"]])
    print("\nShare of comments with red heart per party:")
    print(emoji_comment_share[["party", "has_red_heart_percent"]])
    print("\nShare of comments with pink heart per party:")
    print(emoji_comment_share[["party", "has_pink_heart_percent"]])

def main():
    # configurations
    start_date = config.start_date
    end_date = config.end_date
    folder = f"results/sentiment_analysis/{start_date}_{end_date}_with_emoji_sentiment"
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    topic_dir = os.path.join("results", "topic_analysis", "merged")
    output_sentiment_topic_dir = os.path.join("plots", "sentiment_analysis")

    print("Files found:", files)

    all_comments = []
    other_users = set()


    for file in files:
        user = get_user_from_filename(file)
        party = get_party(user)
        if party == "sonstige":
            other_users.add(user)
            print(f"Unknown party for {user}, skip: {file}")
            continue
        df = pd.read_csv(file)
        # sentiment-mapping and final sentiment calculation
        df["sentiment_num"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
        # map emoji sentiment with tolerance of 0.2
        df["emoji_sentiment"] = df["emoji_sentiment"].map(lambda x: emoji_sentiment_conversion(x) if not pd.isna(x) else np.nan)
        df["match"] = df["sentiment_num"] == df["emoji_sentiment"]
        df["final_sentiment"] = df.apply(compute_final_sentiment, axis=1)
        df["user"] = user
        df["party"] = party
        df["has_blue_heart"] = df["extracted_emojis"].apply(lambda x: has_heart(x, "💙"))
        df["has_green_heart"] = df["extracted_emojis"].apply(lambda x: has_heart(x, "💚"))
        df["has_red_heart"] = df["extracted_emojis"].apply(lambda x: has_heart(x, "❤️"))
        df["has_pink_heart"] = df["extracted_emojis"].apply(lambda x: has_heart(x, "🩷"))
        all_comments.append(df)

    if not all_comments:
        print("No valid comment files found.")
        return

    all_comments_df = pd.concat(all_comments, ignore_index=True)
    all_comments_df_valid = all_comments_df[~all_comments_df["final_sentiment"].isna()]


    # Create plots
    # plot sentiment distributions
    plot_emoji_sentiment_distribution(all_comments_df)
    plot_text_sentiment_distribution(all_comments_df)
    plot_final_sentiment_distribution(all_comments_df_valid)

    # plot blue heart share by party
    plot_blue_heart_share_by_party(all_comments_df)

    # weekly development of sentiment
    plot_weekly_final_sentiment(all_comments_df_valid)

    # get crosstab heatmap of emoji vs text sentiment
    plot_sentiment_crosstab_heatmap(all_comments_df)

    # comparison of emoji sentiment vs text sentiment
    coherence_df = all_comments_df[
    all_comments_df["emoji_sentiment"].isin([-1, 0, 1]) &
    all_comments_df["sentiment_num"].isin([-1, 0, 1])
    ].copy()
    coherence_df["match"] = coherence_df["emoji_sentiment"] == coherence_df["sentiment_num"]

    coherence_rate = coherence_df.groupby("party")["match"].mean()
    print("\nKohärenzrate emoji vs. text pro Partei:")
    print(coherence_rate)


    # comare engagement vs sentiment
    plot_engagement_vs_sentiment(all_comments_df_valid)

   # show emoji usage
    show_emoji_usage(all_comments_df)

    print(f"Number of comments: {len(all_comments_df)}")

    # investigate sentiment by topic
    df_sentiment_topic = merge_comments_with_topics(all_comments_df_valid, topic_dir)
    if df_sentiment_topic is not None:
        analyze_sentiment_by_topic(df_sentiment_topic, output_sentiment_topic_dir)


if __name__ == "__main__":
    main()