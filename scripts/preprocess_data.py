import scripts.config as config
import pandas as pd
import os
from datetime import datetime

usernames = config.usernames

for user in usernames:
    # Video-Daten
    input_video = os.path.join("data", "data_raw", "videos", f"{user}_video_data.csv")
    output_video = os.path.join("data", "data_preprocessed", "videos", f"{user}_video_data_preprocessed.csv")
    try:
        df_video = pd.read_csv(input_video)
    except FileNotFoundError:
        print(f"Video-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")

    try:
        df_video['create_time'] = pd.to_datetime(df_video['create_time'], unit='s')
        df_video.to_csv(output_video, index=False)
    except KeyError:
        print(f"Zeitumwandlung der Video-Daten für {user} fehlgeschlagen.")

    # Kommentar-Daten
    input_comments = os.path.join("data", "data_raw", "comments", f"{user}_comments.csv")
    output_comments = os.path.join("data", "data_raw", "comments", f"{user}_comments_preprocessed.csv")
    try:
        df_comments = pd.read_csv(input_comments)
    except FileNotFoundError:
        print(f"Kommentar-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")
    try:
        df_comments['create_time'] = pd.to_datetime(df_comments['create_time'], unit='s')
        df_comments.to_csv(output_comments, index=False)
    except KeyError:
        print(f"Zeitumwandlung der Kommentar-Daten für {user} fehlgeschlagen.")

    
