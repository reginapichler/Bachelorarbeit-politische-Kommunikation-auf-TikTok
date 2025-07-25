import researchtikpy as rtk
import os
import config_processing as config
import pandas as pd
import time

from dotenv import load_dotenv
from datetime import datetime, timedelta



def str_to_date(d): return datetime.strptime(d, "%Y%m%d")
def date_to_str(d): return d.strftime("%Y%m%d")

def main():
    # configurations
    load_dotenv()
    client_key = os.getenv("CLIENT_KEY")
    client_secret = os.getenv("CLIENT_SECRET")
    start_date = config.start_date
    end_date = config.end_date
    usernames = config.usernames
    # get access token
    token_data = rtk.get_access_token(client_key, client_secret)
    access_token = token_data['access_token']

    # skip users without data
    no_data_file = "data/data_raw/no_data_usernames.txt"
    if os.path.exists(no_data_file):
        with open(no_data_file, "r") as f:
            no_data_usernames = set(line.strip() for line in f if line.strip())
        usernames = [u for u in usernames if u not in no_data_usernames]

    for username in usernames:
        try:
            start_dt = str_to_date(start_date)
            end_dt = str_to_date(end_date)

            # define output paths
            output_videos = os.path.join("data", "data_raw", "videos", f"{username}_video_data_{start_date}_{end_date}.csv")
            output_videos_complete = output_videos.replace(".csv", "_complete.csv")
            output_userinfo = os.path.join("data", "data_raw", "userinfo", f"{username}_userinfo_{start_date}_{end_date}.csv")

            # get userinfo
            if not os.path.exists(output_userinfo):
                user_df = rtk.get_users_info([username], access_token)
                user_df.to_csv(output_userinfo, index=False)
                print(f"Userinfo for {username} saved.")
            else:
                print(f"Userinfo for {username} already exists. Skip userinfo download.")

            # Load videos
            # if complete video data exists, skip download for this user
            if os.path.exists(output_videos_complete):
                print(f"Videos for {username} already complete. Skip video download.")
            else:
                current_dt = start_dt
                found_any = False
                # load video data in batches of 10 days
                while current_dt < end_dt:
                    batch_end_dt = min(current_dt + timedelta(days=9), end_dt)
                    batch_start_str = date_to_str(current_dt)
                    batch_end_str = date_to_str(batch_end_dt)

                    videos_df = rtk.get_videos_info(
                        usernames=[username],
                        access_token=access_token,
                        start_date=batch_start_str,
                        end_date=batch_end_str,
                        max_count=100
                    )

                    if not videos_df.empty:
                        found_any = True
                        # if parts of the video data already exist, append to it and ensure correct column order
                        if os.path.exists(output_videos):
                            existing_cols = pd.read_csv(output_videos, nrows=1).columns.tolist()
                            videos_df = videos_df.reindex(columns=existing_cols)
                            videos_df.to_csv(output_videos, mode='a', header=False, index=False)
                        else:
                            videos_df.to_csv(output_videos, index=False)
                        print(f"No videos for {username} ({batch_start_str}–{batch_end_str}) saved.")
                    else:
                        print(f"No existing videos for {username} in batch {batch_start_str}–{batch_end_str}.")
                    # update current date to next batch (one day after the end of the current batch)
                    current_dt = batch_end_dt + timedelta(days=1)
                    time.sleep(3) # to avoid rate limit issues
                time.sleep(5) # to avoid rate limit issues
                # if all batches were processed, rename the output file to indicate completion
                if os.path.exists(output_videos):
                    os.rename(output_videos, output_videos_complete)
                    print(f"Videos for {username} complete.")
                # Save username to no_data_file if no videos found
                if not found_any:
                    with open(no_data_file, "a") as f:
                        f.write(username + "\n")
                    print(f"{username} added to {no_data_file}.")
                else:
                    # If data was found, remove from no_data_file if username was added there
                    if os.path.exists(no_data_file):
                        with open(no_data_file, "r") as f:
                            lines = f.readlines()
                        with open(no_data_file, "w") as f:
                            for line in lines:
                                if line.strip() != username:
                                    f.write(line)
        except Exception as e:
            print(f"Error at loading videos for {username}: {e}")
            with open(no_data_file, "a") as f:
                f.write(username + "\n")
            continue

if __name__ == "__main__":
    main()
    print("Video download completed.")