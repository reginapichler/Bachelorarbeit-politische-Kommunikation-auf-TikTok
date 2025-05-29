import pandas as pd
import os
import config
import ast

# Verzeichnis mit den Video-CSV-Dateien
video_dir = os.path.join("data", "data_preprocessed", "videos")
results_dir = os.path.join("results", "deskriptive_analyse")
os.makedirs(results_dir, exist_ok=True)

parteien = {
    "Linke": config.linke_usernames,
    "Grüne": config.gruene_usernames,
    "CDUCSU": config.cdu_csu_usernames,
    "AfD": config.afd_usernames,
    "SPD": config.spd_usernames
}
start_date = config.start_date
end_date = config.end_date

no_videos = []

all_dfs = []

for partei, userlist in parteien.items():
    dfs = []
    for user in userlist:
        file_path = os.path.join(video_dir, f"{user}_video_data_preprocessed_{start_date}_{end_date}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Datei nicht gefunden: {file_path}")
            no_videos.append(user)

    if dfs:
        partei_df = pd.concat(dfs, ignore_index=True)
        all_dfs.append(partei_df)

        # Kennzahlen berechnen
        metrics = {
            "Anzahl Videos": len(partei_df),
            "Durchschnittliche Views": partei_df['view_count'].mean(),
            "Durchschnittliche Likes": partei_df['like_count'].mean(),
            "Durchschnittliche Kommentare": partei_df['comment_count'].mean(),
            "Durchschnittliche Shares": partei_df['share_count'].mean()
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(results_dir, f"{partei}_{start_date}_{end_date}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Top 5 Videos
        top_videos = partei_df.nlargest(5, 'view_count')[['id', 'username', 'voice_to_text', 'video_description', 'view_count', 'like_count', 'comment_count', 'share_count']]
        topfive_path = os.path.join(results_dir, f"{partei}_{start_date}_{end_date}_topfive.csv")
        top_videos.to_csv(topfive_path, index=False)

        # Hashtag-Verteilung pro Partei
        for col, label in [("hashtag_names", "hashtags"), ("effect_ids", "effects"), ("playlist_id", "playlists")]:
            if col in partei_df.columns:
                series = partei_df[col].dropna().apply(ast.literal_eval)
                flat = [item for sublist in series for item in sublist if item]
                counts = pd.Series(flat).value_counts().reset_index()
                counts.columns = [label, 'count']
                counts.to_csv(os.path.join(results_dir, f"{partei}_{start_date}_{end_date}_{label}.csv"), index=False)
    else:
        print("Keine Daten für diese Partei gefunden.")

# Verteilungen für alle Parteien zusammen
if all_dfs:
    all_df = pd.concat(all_dfs, ignore_index=True)
    for col, label in [("hashtag_names", "hashtags"), ("effect_ids", "effects"), ("playlist_id", "playlists")]:
        if col in all_df.columns:
            series = all_df[col].dropna().apply(ast.literal_eval)
            flat = [item for sublist in series for item in sublist if item]
            counts = pd.Series(flat).value_counts().reset_index()
            counts.columns = [label, 'count']
            counts.to_csv(os.path.join(results_dir, f"alle_{start_date}_{end_date}_{label}.csv"), index=False)

    # Top 5 Videos für alle Parteien zusammen
    top_videos_all = all_df.nlargest(5, 'view_count')[['id', 'username', 'voice_to_text', 'video_description', 'view_count', 'like_count', 'comment_count', 'share_count']]
    topfive_all_path = os.path.join(results_dir, f"alle_{start_date}_{end_date}_topfive.csv")
    top_videos_all.to_csv(topfive_all_path, index=False)

print(f"Deskriptive Analyse abgeschlossen. Keine Videodaten für: {no_videos}")