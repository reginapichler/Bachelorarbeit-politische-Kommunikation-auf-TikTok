import pandas as pd
import os
import config
import ast
import matplotlib.pyplot as plt

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

def plot_videos_per_party(all_df, plot_dir):
    # 1. Anzahl Videos pro Partei
    videos_per_party = all_df["partei"].value_counts().sort_index()
    plt.figure(figsize=(8,5))
    videos_per_party.plot(kind="bar", color="#1f77b4")
    plt.title("Anzahl Videos pro Partei")
    plt.xlabel("Partei")
    plt.ylabel("Anzahl Videos")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "anzahl_videos_pro_partei.png"))
    plt.close()

def plot_metrics_rate(all_df, plot_dir):
    # 2. Durchschnittliche Views, Likes, Kommentare, Shares pro Partei (Rate)
    metrics = all_df.groupby("partei")[["view_count", "like_count", "comment_count", "share_count"]].mean()
    metrics_rate = metrics.div(metrics["view_count"], axis=0)[["like_count", "comment_count", "share_count"]]
    metrics_rate.plot(kind="bar", figsize=(10,6))
    plt.title("Durchschnittliche Like-, Kommentar- und Share-Rate pro Partei")
    plt.xlabel("Partei")
    plt.ylabel("Rate (pro View)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rate_pro_partei.png"))
    plt.close()

def plot_hashtags(all_df, plot_dir):
    # 3. Verteilung der Hashtags (gesamt)
    if "hashtag_names" in all_df.columns:
        all_hashtags = all_df["hashtag_names"].dropna().apply(safe_literal_eval)
        flat_hashtags = [item for sublist in all_hashtags for item in sublist if item]
        hashtag_counts = pd.Series(flat_hashtags).value_counts().head(20)
        plt.figure(figsize=(10,6))
        hashtag_counts.plot(kind="bar", color="#2ca02c")
        plt.title("Top 20 Hashtags (gesamt)")
        plt.xlabel("Hashtag")
        plt.ylabel("Anzahl")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "hashtags_gesamt.png"))
        plt.close()

        # ...und pro Partei
        for partei in all_df["partei"].unique():
            partei_df = all_df[all_df["partei"] == partei]
            partei_hashtags = partei_df["hashtag_names"].dropna().apply(safe_literal_eval)
            flat_partei_hashtags = [item for sublist in partei_hashtags for item in sublist if item]
            partei_hashtag_counts = pd.Series(flat_partei_hashtags).value_counts().head(10)
            plt.figure(figsize=(8,5))
            partei_hashtag_counts.plot(kind="bar", color="#ff7f0e")
            plt.title(f"Top 10 Hashtags ({partei})")
            plt.xlabel("Hashtag")
            plt.ylabel("Anzahl")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"hashtags_{partei}.png"))
            plt.close()

def plot_time_development(all_df, plot_dir):
    # 4. Zeitliche Entwicklung der Videoanzahl pro Partei (granulare Wochen-Beschriftung)
    if "create_time" in all_df.columns:
        all_df["create_time"] = pd.to_datetime(all_df["create_time"], errors="coerce")
        all_df["woche"] = all_df["create_time"].dt.to_period("W")
        videos_per_month_party = all_df.groupby(["woche", "partei"]).size().unstack(fill_value=0)
        ax = videos_per_month_party.plot(figsize=(12,6))
        plt.title("Zeitliche Entwicklung der Videoanzahl pro Partei")
        plt.xlabel("Woche")
        plt.ylabel("Anzahl Videos")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "zeitliche_entwicklung_videos_pro_partei.png"))
        plt.close()

def plot_videos_per_account(all_df, plot_dir):
    # Plot: Verteilung der Videoanzahl pro Account (gesamt)
    videos_per_account = all_df.groupby("username").size()
    plt.figure(figsize=(10,6))
    videos_per_account.value_counts().sort_index().plot(kind="bar")
    plt.title("Verteilung: Anzahl Videos pro Account (gesamt)")
    plt.xlabel("Anzahl Videos")
    plt.ylabel("Anzahl Accounts")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "anzahl_videos_pro_account.png"))
    plt.close()

def print_account_stats(all_df):
    # Durchschnittliche Anzahl Videos pro Account (gesamt und pro Partei)
    # Gesamt
    videos_per_account = all_df.groupby("username").size()
    avg_videos_per_account = videos_per_account.mean()
    print(f"Durchschnittliche Anzahl Videos pro Account (gesamt): {avg_videos_per_account:.2f}")

    # Pro Partei
    avg_videos_per_account_party = all_df.groupby("partei")["username"].value_counts().groupby("partei").mean()
    print("\nDurchschnittliche Anzahl Videos pro Account pro Partei:")
    print(avg_videos_per_account_party)

    # Anzahl Accounts pro Partei und Anzahl gepostete Videos insgesamt
    active_per_party = all_df.groupby("partei")["username"].nunique()
    total_videos = len(all_df)
    print("\nAnzahl Accounts pro Partei:")
    print(active_per_party)
    print(f"\nAnzahl gepostete Videos insgesamt: {total_videos}")

def main():
    start_date = config.start_date
    end_date = config.end_date
    video_dir = os.path.join("data", "data_preprocessed", "videos")
    comment_dir = os.path.join("data", "data_preprocessed", "comments")
    results_dir = os.path.join("results", "deskriptive_analyse", f"{start_date}_{end_date}")
    plot_dir = os.path.join("plots", "deskriptive_analyse", f"{start_date}_{end_date}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    parteien = {
        "linke": config.linke_usernames,
        "gruene": config.gruene_usernames,
        "cdu_csu": config.cdu_csu_usernames,
        "afd": config.afd_usernames,
        "spd": config.spd_usernames
    }

    no_videos = []
    all_dfs = []

    for partei, userlist in parteien.items():
        partei_df, missing = load_party_data(userlist, video_dir, start_date, end_date)
        no_videos.extend(missing)
        if partei_df is not None:
            partei_df["partei"] = partei
            all_dfs.append(partei_df)
            save_metrics(partei_df, partei, results_dir)
            save_distribution(partei_df, partei, results_dir)
        else:
            print(f"Keine Daten für {partei} gefunden.")

    # Gesamtauswertung
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        save_overall_distribution(all_df, results_dir)

        plot_videos_per_party(all_df, plot_dir)
        plot_metrics_rate(all_df, plot_dir)
        plot_hashtags(all_df, plot_dir)
        plot_time_development(all_df, plot_dir)
        plot_videos_per_account(all_df, plot_dir)

        print_account_stats(all_df)

    # --- Kommentar-Analyse ---
    # Alle Kommentar-CSV-Dateien laden
    comment_files = [os.path.join(comment_dir, f) for f in os.listdir(comment_dir) if f.endswith(".csv")]
    comment_dfs = []
    for f in comment_files:
        try:
            comment_dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Fehler beim Laden von {f}: {e}")
    if not comment_dfs:
        print("Keine Kommentar-Daten gefunden.")
        return
    comments_df = pd.concat(comment_dfs, ignore_index=True)

    # Merge: Partei-Info zu jedem Kommentar holen (über video_id)
    video_party_df = all_df[["id", "partei"]].drop_duplicates()
    comments_merged = comments_df.merge(video_party_df, left_on="video_id", right_on="id", how="left")

    # Rate Kommentare pro Video pro Partei
    comments_per_video = comments_merged.groupby(["partei", "video_id"]).size().reset_index(name="comment_count")
    comments_per_video_party = comments_per_video.groupby("partei")["comment_count"].mean()
    print("\nDurchschnittliche Kommentar-Rate pro Video und Partei:")
    print(comments_per_video_party)

    # Like- und Reply-Rate pro Kommentar für jede Partei
    like_rate_per_party = comments_merged.groupby("partei")["like_count"].mean()
    reply_rate_per_party = comments_merged.groupby("partei")["reply_count"].mean()
    print("\nDurchschnittliche Like-Rate pro Kommentar und Partei:")
    print(like_rate_per_party)
    print("\nDurchschnittliche Reply-Rate pro Kommentar und Partei:")
    print(reply_rate_per_party)

    # Zeitlicher Verlauf der Kommentar-Posts pro Partei (pro Woche)
    comments_merged["create_time"] = pd.to_datetime(comments_merged["create_time"], errors="coerce")
    comments_merged["woche"] = comments_merged["create_time"].dt.to_period("W")
    comments_per_week_party = comments_merged.groupby(["woche", "partei"]).size().unstack(fill_value=0)
    ax = comments_per_week_party.plot(figsize=(12,6))
    plt.title("Zeitlicher Verlauf der Kommentar-Posts pro Partei")
    plt.xlabel("Woche")
    plt.ylabel("Anzahl Kommentare")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "zeitlicher_verlauf_kommentare_pro_partei.png"))
    plt.close()

if __name__ == "__main__":
    main()