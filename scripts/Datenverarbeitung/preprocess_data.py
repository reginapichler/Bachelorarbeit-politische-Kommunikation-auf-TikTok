import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config_processing as config

import pandas as pd
from datetime import datetime
import re
import emoji

def preprocess_video(user, start_date, end_date):
    input_video = os.path.join("data", "data_raw", "videos", f"{user}_video_data_{start_date}_{end_date}_complete.csv")
    output_video = os.path.join("data", "data_preprocessed", "videos", f"{user}_video_data_{start_date}_{end_date}_preprocessed.csv")
    try:
        df_video = pd.read_csv(input_video, engine='python')
    except FileNotFoundError:
        print(f"Video-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")
        return False
    except Exception as e:
        print(f"Fehler bei {user}: {e}")
        return False

    try:
        df_video['create_time'] = pd.to_datetime(df_video['create_time'], unit='s')
        df_video.to_csv(output_video, index=False)
        return True
    except Exception as e:
        print(f"Fehler beim Speichern bei {user}: {e}")
        return False

def clean_text(text):
    # Emojis entfernen
    text = emoji.replace_emoji(str(text), replace='')
    # Satzzeichen entfernen (alles außer Buchstaben, Zahlen und Leerzeichen)
    text = re.sub(r'[^\w\s]', '', text)
    # In Kleinbuchstaben umwandeln
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
        # Text bereinigen
        if 'text' in df.columns:
            df['text'] = df['text'].astype(str).apply(clean_text)
        df.to_csv(output_path, index=False)
    except FileNotFoundError:
        print(f"Kommentar-Daten für {user} nicht gefunden. Verarbeitung übersprungen.")
    except Exception as e:
        print(f"Fehler bei {user}: {e}")

def main():
    usernames = config.usernames
    start_date = config.start_date
    end_date = config.end_date

    no_video = []

    for user in usernames:
        success = preprocess_video(user, start_date, end_date)
        if not success:
            no_video.append(user)

    print("Keine Videos für:")
    for user in no_video:
        print(user)

    comment_cols = [
        "id", "create_time", "text", "like_count", "reply_count",
        "parent_comment_id", "video_id"
    ]

    for user in usernames:
        preprocess_comments(user, start_date, end_date, comment_cols)

if __name__ == "__main__":
    main()
