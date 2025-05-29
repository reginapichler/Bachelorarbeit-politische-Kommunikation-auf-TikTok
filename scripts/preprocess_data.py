import config
import pandas as pd
import os
from datetime import datetime

usernames = config.usernames
start_date = config.start_date
end_date = config.end_date

for user in usernames:
    # Video-Daten
    input_video = os.path.join("data", "data_raw", "videos", f"{user}_video_data_{start_date}_{end_date}_complete.csv")
    output_video = os.path.join("data", "data_preprocessed", "videos", f"{user}_video_data_preprocessed_{start_date}_{end_date}.csv")
    try:
        df_video = pd.read_csv(input_video)
        df_video['create_time'] = pd.to_datetime(df_video['create_time'], unit='s')
        df_video.to_csv(output_video, index=False)
    except FileNotFoundError:
        print(f"Video-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")

    # Kommentar-Daten
    input_comments = os.path.join("data", "data_raw", "comments", f"{user}_comments_{start_date}_{end_date}.csv")
    output_comments = os.path.join("data", "data_preprocessed", "comments", f"{user}_comments_preprocessed_{start_date}_{end_date}.csv")
    try:
        df_comments = pd.read_csv(input_comments)
        df_comments['create_time'] = pd.to_datetime(df_comments['create_time'], unit='s')
        df_comments.to_csv(output_comments, index=False)
    except FileNotFoundError:
        print(f"Kommentar-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")

    
