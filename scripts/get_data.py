import researchtikpy as rtk
import os
import config
from dotenv import load_dotenv

load_dotenv()

client_key = os.getenv("CLIENT_KEY")
client_secret = os.getenv("CLIENT_SECRET")

token_data = rtk.get_access_token(client_key, client_secret)
access_token = token_data['access_token']
start_date = config.start_date
end_date = config.end_date
total_max_count = 100
usernames = config.usernames

for username in usernames:
    # Pfade zu den Ausgabedateien
    output_videos = os.path.join("data", "data_raw", "videos", f"{username}_video_data.csv")
    output_userinfo = os.path.join("data", "data_raw", "userinfo", f"{username}_userinfo.csv")
    output_comments = os.path.join("data", "data_raw", "comments", f"{username}_comments.csv")

    # Video-Daten für den User holen und speichern, falls nicht vorhanden
    if not os.path.exists(output_videos):
        videos_df = rtk.get_videos_info(
            usernames=[username],
            access_token=access_token,
            start_date=start_date,
            end_date=end_date,
            max_count=total_max_count
        )
        videos_df.to_csv(output_videos, index=False)
        print(f"Video-Daten für {username} gespeichert.")
    else:
        videos_df = None
        print(f"Video-Daten für {username} existieren bereits. Überspringe Download.")

    # Userinfo holen und speichern, falls nicht vorhanden
    if not os.path.exists(output_userinfo):
        user_df = rtk.get_users_info([username], access_token)
        user_df.to_csv(output_userinfo, index=False)
        print(f"Userinfo für {username} gespeichert.")
    else:
        print(f"Userinfo für {username} existiert bereits. Überspringe Download.")

    # Kommentare holen und speichern, falls nicht vorhanden
    if not os.path.exists(output_comments):
        # Falls videos_df nicht geladen wurde, lade es jetzt
        if videos_df is None:
            videos_df = rtk.get_videos_info(
                usernames=[username],
                access_token=access_token,
                start_date=start_date,
                end_date=end_date,
                max_count=total_max_count
            )
        comments_df = rtk.get_video_comments(videos_df, access_token, max_count=total_max_count)
        comments_df.to_csv(output_comments, index=False)
        print(f"Kommentare für {username} gespeichert.")
    else:
        print(f"Kommentare für {username} existieren bereits. Überspringe Download.")