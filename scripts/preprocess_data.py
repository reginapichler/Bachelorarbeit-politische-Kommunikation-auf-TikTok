import config
import pandas as pd
import os
from datetime import datetime

usernames = config.usernames
start_date = config.start_date
end_date = config.end_date

no_video = []

for user in usernames:
    input_video = os.path.join("data", "data_raw", "videos", f"{user}_video_data_{start_date}_{end_date}_complete.csv")
    output_video = os.path.join("data", "data_preprocessed", "videos", f"{user}_video_data_preprocessed_{start_date}_{end_date}.csv")
    
    try:
        df_video = pd.read_csv(input_video, engine='python')
    except FileNotFoundError:
        print(f"Video-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")
        no_video.append(user)
        continue
    except Exception as e:
        print(f"Fehler bei {user}: {e}")
        continue

    try:
        # df_video['create_time'] = pd.to_datetime(df_video['create_time'], unit='s')
        df_video.to_csv(output_video, index=False)
    except Exception as e:
        print(f"Fehler beim Speichern bei {user}: {e}")

print("Keine Videos für:")
for user in no_video:
    print(user)
