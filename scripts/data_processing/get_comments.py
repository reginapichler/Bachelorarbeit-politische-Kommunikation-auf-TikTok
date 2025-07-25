import researchtikpy as rtk
import os
import pandas as pd
import time

from dotenv import load_dotenv
from datetime import datetime, timedelta
import config_processing as config


load_dotenv()

# configuration
client_key = os.getenv("CLIENT_KEY")
client_secret = os.getenv("CLIENT_SECRET")

token_data = rtk.get_access_token(client_key, client_secret)
access_token = token_data['access_token']
start_date = config.start_date
end_date = config.end_date
total_max_count = 100
usernames = config.usernames

def str_to_date(d): return datetime.strptime(d, "%Y%m%d")
def date_to_str(d): return d.strftime("%Y%m%d")

def wait_with_exponential_backoff(retries, base_wait=10, max_wait=600):
    wait_time = min(base_wait * (2 ** retries), max_wait)
    print(f"Wait {wait_time} seconds for rate limit (number of retries: {retries+1})...")
    time.sleep(wait_time)

def get_comments_with_retry(batch_videos, access_token, max_count, max_retries=5):
    retries = 0
    while retries <= max_retries:
        try:
            return rtk.get_video_comments(batch_videos, access_token, max_count=max_count)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait_with_exponential_backoff(retries)
                retries += 1
            else:
                raise e
    print("Exceeded maximum retries. Skip.")
    return pd.DataFrame()

def get_no_comments_path(start_date, end_date):
    return os.path.join("data", "data_raw", "comments", f"no_comments_{start_date}_{end_date}.txt")

def load_no_comments_list(start_date, end_date):
    path = get_no_comments_path(start_date, end_date)
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def add_to_no_comments(username, start_date, end_date):
    path = get_no_comments_path(start_date, end_date)
    with open(path, "a") as f:
        f.write(username + "\n")

def get_no_comments_batch_path(batch_start, batch_end):
    return os.path.join("data", "data_raw", "comments", f"no_comments_{batch_start}_{batch_end}.txt")

def load_no_comments_batch_list(batch_start, batch_end):
    path = get_no_comments_batch_path(batch_start, batch_end)
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def add_to_no_comments_batch(username, batch_start, batch_end):
    path = get_no_comments_batch_path(batch_start, batch_end)
    with open(path, "a") as f:
        f.write(username + "\n")


no_comments_users = load_no_comments_list(start_date, end_date)

for username in usernames:
    if username in no_comments_users:
        print(f"{username} included in no_comments.txt. Skipping comment download.")
        continue
    try:
        start_dt = str_to_date(start_date)
        end_dt = str_to_date(end_date)
        output_videos = os.path.join("data", "data_raw", "videos", f"{username}_video_data_{start_date}_{end_date}.csv")
        output_videos_complete = output_videos.replace(".csv", "_complete.csv")
        output_comments = os.path.join("data", "data_raw", "comments", f"{username}_comments_{start_date}_{end_date}.csv")
        output_comments_complete = output_comments.replace(".csv", "_complete.csv")
        # Load comments
        print(f"Load comments for {username}...")
        if os.path.exists(output_comments_complete):
            print(f"Comments for {username} already complete. Skip comment download.")
        else:
            if os.path.exists(output_videos_complete):
                videos_df = pd.read_csv(output_videos_complete)
                current_dt = str_to_date(start_date)
                while current_dt < end_dt:
                    batch_start_str = date_to_str(current_dt)
                    batch_end_dt = min(current_dt + timedelta(days=9), end_dt)
                    batch_end_str = date_to_str(batch_end_dt)

                    # Check whether user is in no_comments_batch
                    no_comments_batch_users = load_no_comments_batch_list(batch_start_str, batch_end_str)
                    if username in no_comments_batch_users:
                        print(f"{username} is in no_comments_{batch_start_str}_{batch_end_str}.txt. Skip batch.")
                        current_dt = batch_end_dt + timedelta(days=1)
                        continue

                    mask = (pd.to_datetime(videos_df['create_time'], unit='s') >= current_dt) & \
                           (pd.to_datetime(videos_df['create_time'], unit='s') <= batch_end_dt)
                    batch_videos = videos_df[mask]

                    # Check whether comments already exist
                    if os.path.exists(output_comments):
                        existing_comments = pd.read_csv(output_comments)
                        already_done_ids = set(existing_comments['parent_comment_id'].dropna().astype(str))
                        batch_videos = batch_videos[~batch_videos['id'].astype(str).isin(already_done_ids)]

                    if not batch_videos.empty:
                        comments_df = get_comments_with_retry(batch_videos, access_token, max_count=total_max_count)
                        time.sleep(3)
                        if not comments_df.empty:
                            if os.path.exists(output_comments):
                                existing_cols = pd.read_csv(output_comments, nrows=1).columns.tolist()
                                comments_df = comments_df.reindex(columns=existing_cols)
                                comments_df.to_csv(output_comments, mode='a', header=False, index=False)
                            else:
                                comments_df.to_csv(output_comments, index=False)
                            print(f"Saved comments for {username} ({batch_start_str}–{batch_end_str})")
                        else:
                            print(f"No comments for {username} in time batch {batch_start_str}–{batch_end_str}.")
                            add_to_no_comments_batch(username, batch_start_str, batch_end_str)
                    else:
                        print(f"All comments for {username} in time batch {batch_start_str}–{batch_end_str} already processed.")

                    current_dt = batch_end_dt + timedelta(days=1)
                if os.path.exists(output_comments):
                    os.rename(output_comments, output_comments_complete)
                    print(f"Comments for {username} complete.")
                    time.sleep(5)
            else:
                print(f"No videos for {username}. Skip download of comments.")
                add_to_no_comments(username, start_date, end_date)
                print(f"{username} added to no_comments.txt (no comments in investigation period).")

    except Exception as e:
        print(f"Error at loading comments for {username}: {e}")
        continue