import researchtikpy as rtk
import os
import config
import pandas as pd
import time

from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

client_key = os.getenv("CLIENT_KEY")
client_secret = os.getenv("CLIENT_SECRET")

token_data = rtk.get_access_token(client_key, client_secret)
access_token = token_data['access_token']
start_date = config.start_date
end_date = config.end_date
total_max_count = 100
usernames = config.usernames

# Hilfsfunktionen
def str_to_date(d): return datetime.strptime(d, "%Y%m%d")
def date_to_str(d): return d.strftime("%Y%m%d")

for username in usernames:
    try:
        start_dt = str_to_date(start_date)
        end_dt = str_to_date(end_date)

        # Dateipfade
        output_videos = os.path.join("data", "data_raw", "videos", f"{username}_video_data_{start_date}_{end_date}.csv")
        output_videos_complete = output_videos.replace(".csv", "_complete.csv")
        output_userinfo = os.path.join("data", "data_raw", "userinfo", f"{username}_userinfo_{start_date}_{end_date}.csv")

        # USERINFO nur einmal holen
        if not os.path.exists(output_userinfo):
            user_df = rtk.get_users_info([username], access_token)
            user_df.to_csv(output_userinfo, index=False)
            print(f"Userinfo für {username} gespeichert.")
        else:
            print(f"Userinfo für {username} existiert bereits. Überspringe Download.")

        # 1. VIDEOS LADEN
        if os.path.exists(output_videos_complete):
            print(f"Videos für {username} bereits komplett vorhanden. Überspringe Video-Download.")
        else:
            current_dt = start_dt
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
                    if os.path.exists(output_videos):
                        videos_df.to_csv(output_videos, mode='a', header=False, index=False)
                    else:
                        videos_df.to_csv(output_videos, index=False)
                    print(f"Video-Daten für {username} ({batch_start_str}–{batch_end_str}) gespeichert.")
                else:
                    print(f"Keine Videos für {username} im Zeitraum {batch_start_str}–{batch_end_str}.")

                current_dt = batch_end_dt + timedelta(days=1)
                time.sleep(3)
            time.sleep(5)
            if os.path.exists(output_videos):
                os.rename(output_videos, output_videos_complete)
                print(f"Video-Datei für {username} komplett.")
    
    except Exception as e:
        print(f"Fehler beim Laden der Videos von {username}: {e}")
        continue