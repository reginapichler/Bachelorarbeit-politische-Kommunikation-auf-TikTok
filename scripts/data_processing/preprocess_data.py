import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config_processing as config

import pandas as pd
from datetime import datetime
import re
import emoji

def get_party(username):
    if username in config.afd_usernames:
        return "afd"
    elif username in config.spd_usernames:
        return "spd"
    elif username in config.cdu_csu_usernames:
        return "cdu_csu"
    elif username in config.gruene_usernames:
        return "grüne"
    elif username in config.linke_usernames:
        return "linke"
    else:
        return "unknown"

def preprocess_video(user, start_date, end_date):
    input_video = os.path.join("data", "data_raw", "videos", f"{user}_video_data_{start_date}_{end_date}_complete.csv")
    output_video = os.path.join("data", "data_preprocessed", "videos", f"{user}_video_data_{start_date}_{end_date}_preprocessed.csv")
    try:
        df_video = pd.read_csv(input_video, engine='python')
    except FileNotFoundError:
        print(f"No video data found for {user}. Skipping preprocessing.")
        return False
    except Exception as e:
        print(f"Error for {user}: {e}")
        return False

    try:
        df_video['create_time'] = pd.to_datetime(df_video['create_time'], unit='s')
        df_video.to_csv(output_video, index=False)
        return True
    except Exception as e:
        print(f"Error at saving for {user}: {e}")
        return False

def clean_text(text):
    # remove emojis
    text = emoji.replace_emoji(str(text), replace='')
    # remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_comments(user, start_date, end_date, comment_cols):
    input_dir = "data/data_raw/comments"
    output_dir = "data/data_preprocessed/comments"
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{user}_comments_{start_date}_{end_date}_complete.csv")
    output_path = os.path.join(output_dir, f"{user}_comments_{start_date}_{end_date}_preprocessed.csv")
    try:
        df = pd.read_csv(input_path, engine='python')
        cols_present = [col for col in comment_cols if col in df.columns]
        df = df[cols_present]
        df['create_time'] = pd.to_datetime(df['create_time'], unit='s', errors='coerce')
        # clean text
        if 'text' in df.columns:
            df['text'] = df['text'].astype(str).apply(clean_text)
        df.to_csv(output_path, index=False)
    except FileNotFoundError:
        print(f"No comments found for {user}. Skipping preprocessing.")
    except Exception as e:
        print(f"Error for {user}: {e}")

def aggregate_and_save_by_party(usernames, start_date, end_date):
    party_video_dfs = {}
    party_comment_dfs = {}

    for user in usernames:
        partei = get_party(user)

        # videos
        video_path = os.path.join("data", "data_preprocessed", "videos", f"{user}_video_data_{start_date}_{end_date}_preprocessed.csv")
        if os.path.exists(video_path):
            df_video = pd.read_csv(video_path)
            df_video["username"] = user
            df_video["partei"] = partei
            party_video_dfs.setdefault(partei, []).append(df_video)

        # comments
        comment_path = os.path.join("data", "data_preprocessed", "comments", f"{user}_comments_{start_date}_{end_date}_preprocessed.csv")
        if os.path.exists(comment_path):
            df_comment = pd.read_csv(comment_path)
            df_comment["username"] = user
            df_comment["partei"] = partei
            party_comment_dfs.setdefault(partei, []).append(df_comment)

    # save
    output_dir = os.path.join("data", "data_preprocessed", "party")
    os.makedirs(output_dir, exist_ok=True)

    for partei, dfs in party_video_dfs.items():
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(os.path.join(output_dir, f"videos_{partei}.csv"), index=False)

    for partei, dfs in party_comment_dfs.items():
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(os.path.join(output_dir, f"comments_{partei}.csv"), index=False)

def main():
    usernames = config.usernames
    start_date = config.start_date
    end_date = config.end_date

    no_video = []

    for user in usernames:
        success = preprocess_video(user, start_date, end_date)
        if not success:
            no_video.append(user)

    print("No videos for:")
    for user in no_video:
        print(user)

    comment_cols = [
        "id", "create_time", "text", "like_count", "reply_count",
        "parent_comment_id", "video_id"
    ]

    for user in usernames:
        preprocess_comments(user, start_date, end_date, comment_cols)

    aggregate_and_save_by_party(usernames, start_date, end_date)

if __name__ == "__main__":
    main()
