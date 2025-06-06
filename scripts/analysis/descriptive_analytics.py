import pandas as pd
import os
import config_analysis as config
import ast

def safe_literal_eval(val):
    """Wertet Strings als Liste aus, gibt sonst leere Liste oder Liste mit Zahl zurück."""
    try:
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, str) and (val.startswith("[") or val.startswith("(")):
            return ast.literal_eval(val)
        return [val]
    except Exception:
        return []

def load_party_data(userlist, video_dir, start_date, end_date):
    """Lädt und verbindet alle Video-CSV-Dateien einer Partei."""
    dfs = []
    missing = []
    for user in userlist:
        file_path = os.path.join(video_dir, f"{user}_video_data_{start_date}_{end_date}_preprocessed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"Datei nicht gefunden: {file_path}")
            missing.append(user)
    if dfs:
        return pd.concat(dfs, ignore_index=True), missing
    else:
        return None, missing

def save_metrics(df, partei, results_dir):
    """Berechnet und speichert Kennzahlen und Top 5 Videos für eine Partei."""
    metrics = {
        "Anzahl Videos": len(df),
        "Durchschnittliche Views": df['view_count'].mean(),
        "Durchschnittliche Likes": df['like_count'].mean(),
        "Durchschnittliche Kommentare": df['comment_count'].mean(),
        "Durchschnittliche Shares": df['share_count'].mean()
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(results_dir, f"{partei}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Top 5 Videos
    top_videos = df.nlargest(5, 'view_count')[['id', 'username', 'voice_to_text', 'video_description', 'view_count', 'like_count', 'comment_count', 'share_count']]
    topfive_path = os.path.join(results_dir, f"{partei}_topfive.csv")
    top_videos.to_csv(topfive_path, index=False)

def save_distribution(df, partei, results_dir):
    """Speichert die Verteilungen von Hashtags, Effekten und Playlists für eine Partei."""
    for col, label in [("hashtag_names", "hashtags"), ("effect_ids", "effects"), ("playlist_id", "playlists"), ("music_id", "music")]:
        if col in df.columns:
            series = df[col].dropna().apply(safe_literal_eval)
            flat = [item for sublist in series for item in sublist if item]
            counts = pd.Series(flat).value_counts().reset_index()
            counts.columns = [label, 'count']
            counts.to_csv(os.path.join(results_dir, f"{partei}_{label}.csv"), index=False)

def save_overall_distribution(all_df, results_dir):
    """Speichert die Verteilungen und Top 5 Videos für alle Parteien zusammen."""
    for col, label in [("hashtag_names", "hashtags"), ("effect_ids", "effects"), ("playlist_id", "playlists"), ("music_id", "music")]:
        if col in all_df.columns:
            series = all_df[col].dropna().apply(safe_literal_eval)
            flat = [item for sublist in series for item in sublist if item]
            counts = pd.Series(flat).value_counts().reset_index()
            counts.columns = [label, 'count']
            counts.to_csv(os.path.join(results_dir, f"alle_{label}.csv"), index=False)

    top_videos_all = all_df.nlargest(5, 'view_count')[['id', 'username', 'voice_to_text', 'video_description', 'view_count', 'like_count', 'comment_count', 'share_count']]
    topfive_all_path = os.path.join(results_dir, f"alle_topfive.csv")
    top_videos_all.to_csv(topfive_all_path, index=False)

def main():
    start_date = config.start_date
    end_date = config.end_date
    video_dir = os.path.join("data", "data_preprocessed", "videos")
    results_dir = os.path.join("results", "deskriptive_analyse", f"{start_date}_{end_date}")
    os.makedirs(results_dir, exist_ok=True)

    parteien = {
        "Linke": config.linke_usernames,
        "Grüne": config.gruene_usernames,
        "CDUCSU": config.cdu_csu_usernames,
        "AfD": config.afd_usernames,
        "SPD": config.spd_usernames
    }

    no_videos = []
    all_dfs = []

    for partei, userlist in parteien.items():
        partei_df, missing = load_party_data(userlist, video_dir, start_date, end_date)
        no_videos.extend(missing)
        if partei_df is not None:
            all_dfs.append(partei_df)
            save_metrics(partei_df, partei, results_dir)
            save_distribution(partei_df, partei, results_dir)
        else:
            print(f"Keine Daten für {partei} gefunden.")

    # Gesamtauswertung
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        save_overall_distribution(all_df, results_dir)

    print(f"Deskriptive Analyse abgeschlossen. Keine Videodaten für: {no_videos}")

if __name__ == "__main__":
    main()