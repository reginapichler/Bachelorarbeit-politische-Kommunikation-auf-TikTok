import glob
import os
import pandas as pd
from collections import Counter
import emoji
import config_analysis as config
import numpy as np


def count_emojis_in_file(filepath):
    """Counts emojis in a CSV file."""
    df = pd.read_csv(filepath, usecols=["text"])
    counter = Counter()
    for text in df["text"].dropna():
        emojis = [e['emoji'] for e in emoji.emoji_list(str(text))]
        counter.update(emojis)
    return counter

def analyze_emojis(input_dir, parteien):
    """Analyzes emojis in comments and counts them per party."""
    emoji_counter = Counter()
    partei_counters = {p: Counter() for p in parteien}

    for fname in os.listdir(input_dir):
        # process only CSV files in the path (those are the comment files)
        if fname.endswith(".csv"):
            # extract username from filename
            user = fname.split("_comments")[0]
            filepath = os.path.join(input_dir, fname)
            file_counter = count_emojis_in_file(filepath)
            emoji_counter.update(file_counter)
            # add emoji count to party the user belongs to
            for partei, userlist in parteien.items():
                if user in userlist:
                    partei_counters[partei].update(file_counter)
    return emoji_counter, partei_counters

def save_emoji_counts(counter, path):
    """Saves emoji counts to a CSV file."""
    df = pd.DataFrame(counter.items(), columns=["emoji", "count"]).sort_values("count", ascending=False)
    df.to_csv(path, index=False)

def resolve_sentiment(row):
    """Chooses sentiment from given scores, gives neutral sentiment is two scores are equal"""
    scores = {
        "negative": row["score_neg"],
        "neutral": row["score_neu"],
        "positive": row["score_pos"]
    }
    max_val = max(scores.values())
    top_labels = [label for label, score in scores.items() if score == max_val]

    if len(top_labels) > 1:
        return "neutral"
    return top_labels[0]

def preprocess_emoji_data(input_path):
    """Preprocesses the emoji sentiment data."""
    emoji_data = pd.read_csv(input_path, sep=";")
    # compute shares of positive, negative, neutral as score
    emoji_data['N'] = emoji_data['Negative'] + emoji_data['Neutral'] + emoji_data['Positive']
    emoji_data['score_pos'] = emoji_data['Positive'] / emoji_data['N']
    emoji_data['score_neg'] = emoji_data['Negative'] / emoji_data['N']
    emoji_data['score_neu'] = emoji_data['Neutral'] / emoji_data['N']

    # sentiment label based on maximum share of positive, negative, neutral
    # sentiment_label: 1 = positive, 0 = neutral, -1 = negative
    emoji_data['sentiment_label_str'] = emoji_data.apply(resolve_sentiment, axis=1)

    emoji_data['sentiment_label'] = emoji_data['sentiment_label_str'].map({
        'negative': -1,
        'neutral': 0,
        'positive': 1
    })

    return emoji_data

def compute_emoji_sentiment(input_folder_sentiment, input_dir, emoji_df, folder_output):
    """Computes emoji sentiment for each comment and saves the results."""
    os.makedirs(folder_output, exist_ok=True)
    sentiment_files = glob.glob(input_folder_sentiment)
    # load sentiment data
    for file in sentiment_files:
        base = os.path.basename(file)
        user = base.split("_comments")[0]
        # get original comment data
        orig_file = [f for f in os.listdir(input_dir) if f.startswith(user) and f.endswith(".csv")]
        if not orig_file:
            print(f"No original comment data for {user} found.")
            continue
        orig_path = os.path.join(input_dir, orig_file[0])
        df_orig = pd.read_csv(orig_path)
        df_sent = pd.read_csv(file)
        # get emojis of original comments
        df_orig['emojis'] = df_orig['text'].apply(lambda x: [e['emoji'] for e in emoji.emoji_list(str(x))])

        
        emoji_sentiments = []
        extracted_emojis = []
        for idx, row in df_sent.iterrows():
            comment_id = row['id']
            orig_row = df_orig[df_orig['id'] == comment_id]
            if orig_row.empty:
                emoji_sentiments.append(np.nan)
                extracted_emojis.append([])
                continue
            # get emojis from original comment
            emojis = orig_row.iloc[0]['emojis']
            extracted_emojis.append(emojis)
            if not emojis:
                emoji_sentiments.append(np.nan)
                continue
            sentiments = []
            # get emoji sentiment for each emoji in the comment
            for em in emojis:
                s = emoji_df.loc[emoji_df['Emoji'] == em, 'sentiment_label']
                if not s.empty:
                    sentiments.append(s.values[0])
                else:
                    sentiments.append(np.nan)
            # compute mean sentiment for emojis in comment
            if sentiments and not all(pd.isna(s) for s in sentiments):
                emoji_sentiments.append(np.nanmean(sentiments))
            else:
                emoji_sentiments.append(np.nan)
        df_sent['emoji_sentiment'] = emoji_sentiments
        df_sent['extracted_emojis'] = extracted_emojis
        # Save
        outname = base.replace('_complete.csv', '_final.csv')
        output_path = os.path.join(folder_output, outname)
        df_sent.to_csv(output_path, index=False)
        print(f"Emoji-Sentiment für {file} gespeichert unter: {output_path}")

def main():
    # configurations
    start_date = config.start_date
    end_date = config.end_date
    input_dir = "data/data_raw/comments"
    input_path_emoji = "data/data_raw/emoji_sentiment_data.csv"
    input_folder_sentiment = f"results/sentiment_analysis/{start_date}_{end_date}/*.csv"
    output_folder_sentiment = f"results/sentiment_analysis/{start_date}_{end_date}_with_emoji_sentiment"
    output_dir = f"results/emoji_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "emoji_counts.csv")

    parteien = {
        "afd": config.afd_usernames,
        "spd": config.spd_usernames,
        "cducsu": config.cdu_csu_usernames,
        "gruene": config.gruene_usernames,
        "linke": config.linke_usernames
    }
    # Analyze emojis in comments
    emoji_counter, partei_counters = analyze_emojis(input_dir, parteien)

    # Save data complete
    save_emoji_counts(emoji_counter, output_path)
    print(f"Emoji-list (complete) saved at: {output_path}")

    # Save per party
    for partei, counter in partei_counters.items():
        partei_path = os.path.join(output_dir, f"emoji_counts_{partei}.csv")
        save_emoji_counts(counter, partei_path)
        print(f"Emoji-list ({partei}) saved at: {partei_path}")

    # Process extracted emoji sentiment data
    emoji_df = preprocess_emoji_data(input_path_emoji)

    print("Unique vals of sentiment label", emoji_df['sentiment_label'].unique())
    compute_emoji_sentiment(input_folder_sentiment, input_dir, emoji_df, output_folder_sentiment)

if __name__ == "__main__":
    main()