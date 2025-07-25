import pandas as pd
import os
import config
import ast
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16
})

def safe_literal_eval(val):
    """Safely evaluates strings as lists, returns empty list or list with number otherwise."""
    try:
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, str) and (val.startswith("[") or val.startswith("(")):
            return ast.literal_eval(val)
        return [val]
    except Exception:
        return []

def load_party_data(userlist, video_dir, start_date, end_date):
    """Loads and concatenates all video CSV files for a party."""
    dfs = []
    missing = []
    for user in userlist:
        file_path = os.path.join(video_dir, f"{user}_video_data_{start_date}_{end_date}_preprocessed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"File not found: {file_path}")
            missing.append(user)
    if dfs:
        return pd.concat(dfs, ignore_index=True), missing
    else:
        return None, missing

def save_metrics(df, party, results_dir):
    """Calculates and saves metrics and top 5 videos for a party."""
    metrics = {
        "Number of videos": len(df),
        "Average views": df['view_count'].mean(),
        "Average likes": df['like_count'].mean(),
        "Average comments": df['comment_count'].mean(),
        "Average shares": df['share_count'].mean(),
        "Total views": df["view_count"].sum()
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(results_dir, f"{party}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics for {party} saved, total views: {metrics['Total views']}")
    print(f"Number of videos for {party}: {len(df)}")
    # Top 5 videos
    top_videos = df.nlargest(5, 'view_count')[['id', 'username', 'voice_to_text', 'video_description', 'view_count', 'like_count', 'comment_count', 'share_count']]
    topfive_path = os.path.join(results_dir, f"{party}_topfive.csv")
    top_videos.to_csv(topfive_path, index=False)

def save_distribution(df, party, results_dir):
    """Saves the distributions of hashtags, effects, playlists, and music for a party."""
    for col, label in [("hashtag_names", "hashtags"), ("effect_ids", "effects"), ("playlist_id", "playlists"), ("music_id", "music")]:
        if col in df.columns:
            series = df[col].dropna().apply(safe_literal_eval)
            flat = [item for sublist in series for item in sublist if item]
            counts = pd.Series(flat).value_counts().reset_index()
            counts.columns = [label, 'count']
            counts.to_csv(os.path.join(results_dir, f"{party}_{label}.csv"), index=False)

def save_overall_distribution(all_df, results_dir):
    """Saves the distributions and top 5 videos for all parties combined."""
    for col, label in [("hashtag_names", "hashtags"), ("effect_ids", "effects"), ("playlist_id", "playlists"), ("music_id", "music")]:
        if col in all_df.columns:
            series = all_df[col].dropna().apply(safe_literal_eval)
            flat = [item for sublist in series for item in sublist if item]
            counts = pd.Series(flat).value_counts().reset_index()
            counts.columns = [label, 'count']
            counts.to_csv(os.path.join(results_dir, f"all_{label}.csv"), index=False)

    top_videos_all = all_df.nlargest(5, 'view_count')[['id', 'username', 'voice_to_text', 'video_description', 'view_count', 'like_count', 'comment_count', 'share_count']]
    topfive_all_path = os.path.join(results_dir, f"all_topfive.csv")
    top_videos_all.to_csv(topfive_all_path, index=False)

def plot_videos_per_party(all_df, plot_dir):
    """Plots the number of videos per party."""
    videos_per_party = all_df["party"].value_counts().sort_index()
    print("Number of videos per party:")
    for party, count in videos_per_party.items():
        print(f"{party}: {count}")
    plt.figure(figsize=(8,5))
    videos_per_party.plot(kind="bar", color="#1f77b4")
    plt.title("Number of videos per party", fontweight="bold", pad=15)
    plt.xlabel("Party")
    plt.ylabel("Number of videos")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "number_of_videos_per_party.png"), dpi=300)
    plt.close()

def plot_metrics_rate(all_df, plot_dir):
    """Plots the average like, comment, and share rates per view for each party."""
    parties = ["afd", "cdu_csu", "gruene", "linke", "spd"]
    labels = ["AfD", "CDU/CSU", "Grüne", "Linke", "SPD"]

    # get metrics per party
    metrics = all_df.groupby("party")[["view_count", "like_count", "comment_count", "share_count"]].mean()
    metrics_rate = metrics.div(metrics["view_count"], axis=0)[["like_count", "comment_count", "share_count"]]
    metrics_rate = metrics_rate.reindex(parties)

    # convert to percentage
    metrics_rate = metrics_rate * 100

    print("Mean rate per view:")
    for party in parties:
        print(f"{party}: {metrics_rate.loc[party].to_dict()}")

    colors = ["#D566A3", "#F0E442", "#00189E"]

    # create bar plot
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    metrics_rate.plot(kind="bar", ax=ax, color=colors, fontsize=16)

    ax.set_title(
        "Durchschnittliche Like-, Kommentar- und Weiterleitungs-Rate pro Partei",
        fontsize=16, pad=15, fontweight="bold"
    )
    ax.set_xlabel("Partei", fontsize=16)
    ax.set_ylabel("Rate pro Ansicht (%)", fontsize=16)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)

    ax.legend(["Likes", "Kommentare", "Weiterleitungen"], fontsize=16)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(plot_dir, "rate_pro_partei.png"), dpi=300)
    plt.close()

def plot_hashtags(all_df, plot_dir):
    """Plots the top hashtags for all parties and overall."""

    # get top hashtags and plot
    if "hashtag_names" in all_df.columns:
        all_hashtags = all_df["hashtag_names"].dropna().apply(safe_literal_eval)
        flat_hashtags = [item for sublist in all_hashtags for item in sublist if item]
        hashtag_counts = pd.Series(flat_hashtags).value_counts().head(20)
        plt.figure(figsize=(10,6))
        hashtag_counts.plot(kind="bar", color="#2ca02c")
        plt.title("Top 20 hashtags (all)", fontweight="bold", pad=15)
        plt.xlabel("Hashtag")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "hashtags_all.png"), dpi=300)
        plt.close()

        # per party
        for party in all_df["party"].unique():
            party_df = all_df[all_df["party"] == party]
            party_hashtags = party_df["hashtag_names"].dropna().apply(safe_literal_eval)
            flat_party_hashtags = [item for sublist in party_hashtags for item in sublist if item]
            party_hashtag_counts = pd.Series(flat_party_hashtags).value_counts().head(10)
            plt.figure(figsize=(8,5))
            party_hashtag_counts.plot(kind="bar", color="#ff7f0e")
            plt.title(f"Top 10 hashtags ({party})", fontweight="bold", pad=15)
            plt.xlabel("Hashtag")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"hashtags_{party}.png"), dpi=300)
            plt.close()

def plot_time_development(all_df, plot_dir):
    """Plots the time development of postings per party."""

    # convert create_time and filter from 2025-01-01
    all_df["create_time"] = pd.to_datetime(all_df["create_time"], errors="coerce")
    all_df = all_df[all_df["create_time"] >= "2025-01-01"]
    all_df["week"] = all_df["create_time"].dt.to_period("W")

    # Group by week & party
    videos_per_week_party = all_df.groupby(["week", "party"]).size().unstack(fill_value=0)
    videos_per_week_party.index = videos_per_week_party.index.to_timestamp()

    parties = ["afd", "cdu_csu", "gruene", "linke", "spd"]
    colors = {
        "afd": "#56B4E9",
        "cdu_csu": "#000000",
        "gruene": "#00B140",
        "linke": "#E10098",
        "spd": "#D00000"
    }
    legend_labels = {
        "afd": "AfD",
        "cdu_csu": "CDU/CSU",
        "gruene": "Grüne",
        "linke": "Linke",
        "spd": "SPD"
    }

    # create the labels for x-axis
    xtick_labels = [d.strftime("%d.%m.%y") if d >= pd.Timestamp("2025-01-01") else "" for d in videos_per_week_party.index]

    # create plot
    ax = videos_per_week_party[parties].plot(
        figsize=(10, 6),
        color=[colors[p] for p in parties],
        linewidth=2
    )

    plt.title("Zeitliche Entwicklung der Videoanzahl pro Partei", fontsize=18, pad=15, fontweight="bold")
    plt.xlabel("Woche", fontsize=18)
    plt.ylabel("Anzahl Videos", fontsize=18)

    ax.set_xticks(videos_per_week_party.index)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=18)

    ax.legend([legend_labels[p] for p in parties], fontsize=16, title="Partei", title_fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "zeitliche_entwicklung_videos_pro_partei.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_weekly_comments(df, output_path):
    """Plots the weekly development of comments per party."""
    
    # convert create_time and filter from 2025-01-01
    df["create_time"] = pd.to_datetime(df["create_time"], errors="coerce")
    df = df[df["create_time"] >= "2025-01-01"]
    df["week"] = df["create_time"].dt.to_period("W")
    grouped = df.groupby(["week", "party"]).size().unstack(fill_value=0)
    grouped.index = grouped.index.to_timestamp()

    parties = ["afd", "cdu_csu", "gruene", "linke", "spd"]
    colors = {
        "afd": "#56B4E9", "cdu_csu": "#000000", "gruene": "#00B140",
        "linke": "#E10098", "spd": "#D00000"
    }
    legend_labels = {
        "afd": "AfD", "cdu_csu": "CDU/CSU", "gruene": "Grüne",
        "linke": "Linke", "spd": "SPD"
    }

    # create the labels for x-axis
    xtick_labels = [d.strftime("%d.%m.%y") if d >= pd.Timestamp("2025-01-01") else "" for d in grouped.index]

    ax = grouped[parties].plot(
        figsize=(10, 6),
        color=[colors[p] for p in parties],
        linewidth=2
    )
    plt.title("Zeitlicher Verlauf der Kommentar-Veröffentlichungen pro Partei", fontsize=18, pad=15, fontweight="bold")
    plt.xlabel("Woche", fontsize=18)
    plt.ylabel("Anzahl Kommentare", fontsize=18)
    ax.set_xticks(grouped.index)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=18)
    ax.legend([legend_labels[p] for p in parties], fontsize=16, title="Partei", title_fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "zeitlicher_verlauf_kommentare_pro_partei.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_videos_per_account(all_df, plot_dir):
    """Plots the distribution of videos per account."""
    # get number of videos per account
    videos_per_account = all_df.groupby("username").size()
    plt.figure(figsize=(10,6))
    videos_per_account.value_counts().sort_index().plot(kind="bar")
    plt.title("Verteilung: Anzahl Videos pro Account (gesamt)", fontweight="bold", pad=15)
    plt.xlabel("Anzahl Videos")
    plt.ylabel("Anzahl Accounts")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "anzahl_videos_pro_account.png"), dpi=300)
    plt.close()

def save_account_stats(all_df, results_dir):
    """Saves statistics about accounts per party."""
    # get users
    party_userlist = {
        "linke": set(config.linke_usernames),
        "gruene": set(config.gruene_usernames),
        "cdu_csu": set(config.cdu_csu_usernames),
        "afd": set(config.afd_usernames),
        "spd": set(config.spd_usernames)
    }

    # df for all parties
    parties = list(party_userlist.keys())
    rows = []

    for party in parties:
        userlist = party_userlist[party]
        party_df = all_df[(all_df["party"] == party) & (all_df["username"].isin(userlist))]
        # Average number of videos per account and active accounts
        if not party_df.empty:
            videos_per_account = party_df.groupby("username").size()
            avg_videos_per_account = videos_per_account.mean()
            active_accounts = videos_per_account.count()
        else:
            avg_videos_per_account = 0
            active_accounts = 0
        accounts_in_config = len(userlist)
        rows.append({
            "party": party,
            "avg_videos_per_account": avg_videos_per_account,
            "active_accounts": active_accounts,
            "accounts_in_config": accounts_in_config
        })
        print(f"{party}: {active_accounts} active accounts")

    # Complete vals for all parties
    all_usernames = set().union(*party_userlist.values())
    all_df_config = all_df[all_df["username"].isin(all_usernames)]
    videos_per_account_all = all_df_config.groupby("username").size()
    avg_videos_per_account_all = videos_per_account_all.mean()
    active_accounts_all = videos_per_account_all.count()
    accounts_in_config_all = len(all_usernames)

    rows.append({
        "party": "complete",
        "avg_videos_per_account": avg_videos_per_account_all,
        "active_accounts": active_accounts_all,
        "accounts_in_config": accounts_in_config_all
    })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(results_dir, "account_stats_alle_parteien.csv"), index=False)

def main():
    # configurations
    start_date = config.start_date
    end_date = config.end_date
    video_dir = os.path.join("data", "data_preprocessed", "videos")
    comment_dir = os.path.join("data", "data_preprocessed", "comments")
    results_dir = os.path.join("results", "deskriptive_analyse")
    plot_dir = os.path.join("plots", "deskriptive_analyse")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # get usernames from config
    parties = {
        "linke": config.linke_usernames,
        "gruene": config.gruene_usernames,
        "cdu_csu": config.cdu_csu_usernames,
        "afd": config.afd_usernames,
        "spd": config.spd_usernames
    }

    no_videos = []
    all_dfs = []

    # Load and process video data for each party
    for party, userlist in parties.items():
        party_df, missing = load_party_data(userlist, video_dir, start_date, end_date)
        no_videos.extend(missing)
        if party_df is not None:
            party_df["party"] = party
            all_dfs.append(party_df)
            save_metrics(party_df, party, results_dir)
            save_distribution(party_df, party, results_dir)
        else:
            print(f"No data found for {party}")

    # Complete evaluation
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        save_overall_distribution(all_df, results_dir)

        plot_videos_per_party(all_df, plot_dir)
        plot_metrics_rate(all_df, plot_dir)
        plot_hashtags(all_df, plot_dir)
        plot_time_development(all_df, plot_dir)
        plot_videos_per_account(all_df, plot_dir)

        save_account_stats(all_df, results_dir)

    # Comment analysis
    # get all comments
    comment_files = [os.path.join(comment_dir, f) for f in os.listdir(comment_dir) if f.endswith(".csv")]
    comment_dfs = []
    for f in comment_files:
        try:
            comment_dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error at loading {f}: {e}")
    if not comment_dfs:
        print("No comment data found.")
        return
    comments_df = pd.concat(comment_dfs, ignore_index=True)

    # Merge: get party info from video data
    video_party_df = all_df[["id", "party"]].drop_duplicates()
    comments_merged = comments_df.merge(video_party_df, left_on="video_id", right_on="id", how="left")

    # Comment rate per video and party
    comments_per_video = comments_merged.groupby(["party", "video_id"]).size().reset_index(name="comment_count")
    comments_per_video_party = comments_per_video.groupby("party")["comment_count"].mean()
    print("\nAverage comments per video per party:")
    print(comments_per_video_party)

    # Like and reply rates for each party
    like_rate_per_party = comments_merged.groupby("party")["like_count"].mean()
    reply_rate_per_party = comments_merged.groupby("party")["reply_count"].mean()
    print("\nAverage like-rate per party and comment:")
    print(like_rate_per_party)
    print("\nAverage reply-rate by comment and party:")
    print(reply_rate_per_party)

    # plot time development of comments
    plot_weekly_comments(comments_merged, plot_dir)

    # Save comment statistics
    stats_df = pd.DataFrame({
        "party": comments_per_video_party.index,
        "avg_comments_per_video": comments_per_video_party.values,
        "avg_likes_per_comment": like_rate_per_party.reindex(comments_per_video_party.index).values,
        "avg_replies_per_comment": reply_rate_per_party.reindex(comments_per_video_party.index).values
    })

    # Save combined comment stats for all parties
    stats_df.to_csv(os.path.join(results_dir, "kommentar_stats_alle_parteien.csv"), index=False)

if __name__ == "__main__":
    main()