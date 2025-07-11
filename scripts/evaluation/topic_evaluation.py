import os
import pandas as pd
import re
import ast
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

# configurations
input_dir = "results/topic_analysis/20250101_20250223"
input_labeled_dir = "results/topic_analysis/20250101_20250223/cleaned/subset_labeled"
clean_dir = os.path.join(input_dir, "cleaned")
plot_dir = os.path.join("plots", "topic_analysis")
preprocessed_dir = os.path.join("data", "data_preprocessed", "party")
os.makedirs(clean_dir, exist_ok=True)

files = [
    "AfD.csv",
    "CDU_CSU.csv",
    "Grüne.csv",
    "SPD.csv",
    "Linke.csv"
]

# mapping for topic categories
topic_keywords = {
    "Soziales": "Soziales & Arbeit",
    "Arbeit": "Soziales & Arbeit",
    "Wirtschaft": "Wirtschaft & Finanzen",
    "Finanzen": "Wirtschaft & Finanzen",
    "Sicherheit": "Sicherheit & Ordnung",
    "Ordnung": "Sicherheit & Ordnung",
    "Migration": "Migration",
    "Umwelt": "Umwelt & Energie",
    "Energie": "Umwelt & Energie",
    "International": "Internationale Politik",
    "Internationale": "Internationale Politik",
    "Persönliches": "Persönliches",
    "Wahlkampf": "Wahlkampf"}

# Regex for topic extraction
keyword_pattern = "|".join(re.escape(k) for k in topic_keywords.keys())

def map_topics_short(text):
    mapping = {
        "S&A": "Soziales & Arbeit",
        "W&F": "Wirtschaft & Finanzen",
        "S&O": "Sicherheit & Ordnung",
        "M": "Migration",
        "U&E": "Umwelt & Energie",
        "I": "Internationale Politik",
        "P": "Persönliches",
        "W": "Wahlkampf",
    }
    return mapping.get(str(text).strip(), text)

def read_labeled_data(input_labeled_dir, files):
    labeled_data = {}
    for filename in files:
        path = os.path.join(input_labeled_dir, filename)
        out_path = os.path.join(input_labeled_dir, "processed", filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=None, engine="python", quoting=0)
                df["topic_clean"] = df["gpt_topic"].apply(map_topics_short)
                df["topic_clean"] = df["topic_clean"].apply(lambda x: [x] if isinstance(x, str) else x)
                labeled_data[filename] = df
                df.to_csv(out_path, index=False)
            except Exception as e:
                print(f"Error reading labeled file {path}: {e}")
    return labeled_data

def use_labeled_if_empty(row):
    orig = row["topic_clean"]
    labeled = row["topic_clean_labeled"]

    # beide als Liste parsen
    if isinstance(orig, str):
        try:
            orig = ast.literal_eval(orig)
        except:
            orig = []
    if not isinstance(orig, list):
        orig = []

    if isinstance(labeled, str):
        try:
            labeled = ast.literal_eval(labeled)
        except:
            labeled = []
    if not isinstance(labeled, list):
        labeled = []

    return labeled if not orig and labeled else orig

def extract_topics(text):
    if pd.isna(text):
        return []
    found = set()
    for match in re.findall(keyword_pattern, text):
        found.add(topic_keywords[match])
    return sorted(found)


def process_and_save_files(dataframes, clean_dir):
    try:
        for filename, df in dataframes.items():
            out_path = os.path.join(clean_dir, filename)
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error saving files: {e}")


def count_topic_lengths(dataframes):
    for filename, df in dataframes.items():
        df["topic_clean"] = df["topic_clean"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        topic_counts = df["topic_clean"].apply(len)
        count_summary = topic_counts.value_counts().sort_index()
        print(f"\n{filename}:")
        for n, count in count_summary.items():
            print(f"{n} unique topics: in {count} rows")


def save_rows_without_topic(dataframes, base_dir):
    subset_dir = os.path.join(base_dir, "subset")
    os.makedirs(subset_dir, exist_ok=True)
    cleaned_dataframes = {}

    for filename, df in dataframes.items():
        df["topic_clean"] = df["topic_clean"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_no_topic = df[df["topic_clean"].apply(lambda x: len(x) == 0)]
        df_with_topic = df[df["topic_clean"].apply(lambda x: len(x) > 0)]

        if not df_no_topic.empty:
            out_path = os.path.join(subset_dir, filename)
            df_no_topic.to_csv(out_path, index=False)
            print(f"{len(df_no_topic)} rows without topic for {filename} saved in: {out_path}")

        print(f"Rows without topic: {len(df_no_topic)} in {filename}")
        cleaned_dataframes[filename] = df_with_topic

    return cleaned_dataframes


def plot_topic_distribution(dataframes, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    for filename, df in dataframes.items():
        df["topic_clean"] = df["topic_clean"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df = df[df["topic_clean"].apply(lambda x: len(x) > 0)]
        if df.empty:
            print(f"No rows with topic in {filename}")
            continue

        all_topics = [topic for topics in df["topic_clean"] for topic in topics]
        topic_counts = pd.Series(all_topics).value_counts(normalize=True).sort_values(ascending=False) * 100

        print(f"\nShare of topics {filename} (percent):")
        print(topic_counts.round(2))

        # Plot
        plt.figure(figsize=(8, 4))
        topic_counts.plot(kind="bar")
        plt.title(f"Topic-Verteilung (Anteile): {filename.replace('.csv', '')}")
        plt.ylabel("Anteil (%)")
        plt.xlabel("Topic")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{filename.replace('.csv', '')}_topic_distribution_share.png"))
        plt.close()



def remove_wahlkampf(dataframes, output_dir):
    try:
        cleaned_dataframes = {}
        removed_dir = os.path.join(output_dir, "wahlkampf_removed_rows")
        os.makedirs(removed_dir, exist_ok=True)

        for filename, df in dataframes.items():
            df = df.copy()

            # remove "Wahlkampf" from topic_clean
            df["topic_clean"] = df["topic_clean"].apply(
                lambda x: [t for t in (ast.literal_eval(x) if isinstance(x, str) else x) if t != "Wahlkampf"]
            )

            # Save empty rows
            df_empty = df[df["topic_clean"].apply(lambda x: len(x) == 0)]
            df_cleaned = df[df["topic_clean"].apply(lambda x: len(x) > 0)]

            if not df_empty.empty:
                out_path = os.path.join(removed_dir, filename)
                df_empty.to_csv(out_path, index=False)
                print(f"{len(df_empty)} rows without Wahlkampf topic saved in: {out_path}")

            cleaned_dataframes[filename] = df_cleaned

        return cleaned_dataframes
    except Exception as e:
        print(f"Error removing Wahlkampf topic: {e}")

def merge_original_data(dataframes, preprocessed_dir):
    merged_dataframes = {}

    for filename, df_topics in dataframes.items():
        partei = filename.replace(".csv", "")
        video_path = os.path.join(preprocessed_dir, f"videos_{partei}.csv")

        if not os.path.exists(video_path):
            print(f"Video data not found for {partei}: {video_path}")
            continue

        try:
            df_videos = pd.read_csv(video_path)
        except Exception as e:
            print(f"Couldn't read {video_path}: {e}")
            continue

        merge_keys = ["id", "username", "video_description", "voice_to_text"]

        df_merged = pd.merge(
            df_topics,
            df_videos,
            on=merge_keys,
            how="left"
        )
        merged_dataframes[filename] = df_merged

        # check merge results
        duplicates = df_merged.duplicated(subset=merge_keys).sum()
        missing_create_time = df_merged["create_time"].isna().sum()

        print(f"len before: {len(df_topics)}, len after merge: {len(df_merged)} for {filename}")
        print(f"Duplicates found in {filename}: {duplicates}")
        print(f"Missing create_time in {filename}: {missing_create_time}")

    return merged_dataframes


def plot_topic_timeline(merged_dataframes, freq="W"):
    all_topics = sorted([
        "Internationale Politik",
        "Migration",
        "Persönliches",
        "Sicherheit & Ordnung",
        "Soziales & Arbeit",
        "Umwelt & Energie",
        "Wahlkampf",
        "Wirtschaft & Finanzen"
    ])
    color_list = plt.cm.get_cmap("tab10", len(all_topics)).colors
    topic_colors = dict(zip(all_topics, color_list))

    for filename, df in merged_dataframes.items():
        partei = filename.replace(".csv", "")
        df = df.copy()

        df["create_time"] = pd.to_datetime(df["create_time"], errors="coerce")
        df = df.dropna(subset=["create_time", "topic_clean"])

        df["topic_clean"] = df["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df = df.explode("topic_clean")
        df["period"] = df["create_time"].dt.to_period(freq).dt.to_timestamp()

        grouped = df.groupby(["period", "topic_clean"]).size().unstack(fill_value=0)

        # choose timeframe
        # TODO: necessary?
        end_date = pd.Timestamp("2025-02-23")
        grouped = grouped.loc[:end_date]
        grouped = grouped.reindex(columns=all_topics, fill_value=0)

        # compute percentages
        grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        grouped.plot(
            ax=ax,
            color=[topic_colors[col] for col in grouped.columns]
        )
        ax.set_title(f"Zeitlicher Verlauf der Topics (Anteile) – {partei}")
        ax.set_xlabel("Zeitraum")
        ax.set_ylabel("Anteil an geposteten Videos (%)")
        ax.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"plots/topic_analysis/topic_timeline_{partei}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved timeline for {partei}")

def calculate_engagement_metrics(merged_dataframes, output_dir):
    all_data = []

    for filename, df in merged_dataframes.items():
        partei = filename.replace(".csv", "")
        df = df.copy()

        # preparation
        df["topic_clean"] = df["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df = df.explode("topic_clean")
        df = df.dropna(subset=["topic_clean"])

        df["partei"] = partei
        all_data.append(df)

    # combine everything in a big df
    combined_df = pd.concat(all_data, ignore_index=True)

    metrics = ["like_count", "view_count", "share_count", "comment_count"]

    overall = combined_df.groupby("topic_clean")[metrics].mean().round(2)

    by_party = combined_df.groupby(["partei", "topic_clean"])[metrics].mean().round(2)

    # Save as csv
    os.makedirs(output_dir, exist_ok=True)
    overall.to_csv(os.path.join(output_dir, "engagement_by_topic.csv"))
    by_party.to_csv(os.path.join(output_dir, "engagement_by_party_topic.csv"))

    print("Engagement-Metriken (gesamt):")
    print(overall)

    print("Engagement-Metriken pro Partei:")
    print(by_party)

    return overall, by_party

def plot_engagement_overall(overall_df, output_dir):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    ax = overall_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Durchschnittliche Engagement-Metriken pro Topic (gesamt)")
    plt.ylabel("Durchschnitt pro Video")
    plt.xlabel("Topic")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engagement_by_topic.png"))
    plt.close()
    print("Saved: engagement_by_topic.png")

def plot_engagement_by_party(by_party_df, output_dir):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    for metric in by_party_df.columns:
        pivot = by_party_df.reset_index().pivot(index="topic_clean", columns="partei", values=metric)
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        plt.title(f"Durchschnittlicher {metric.replace('_', ' ').capitalize()} pro Topic & Partei")
        plt.ylabel("Durchschnitt pro Video")
        plt.xlabel("Topic")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Partei")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_party.png"))
        plt.close()
        print(f"Saved plot: {metric}_by_party.png")

def analyze_topic_combinations(dataframes):
    for filename, df in dataframes.items():
        print(f"\n--- Analysis for: {filename} ---")
        
        df["topic_clean"] = df["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        total_entries = len(df)

        # Count entries without any topic
        num_no_topic = df["topic_clean"].apply(lambda x: len(x) == 0).sum()
        share_no_topic = (num_no_topic / total_entries) * 100 if total_entries > 0 else 0

        print(f"Total entries: {total_entries}")
        print(f"Number of entries without topic: {num_no_topic} ({share_no_topic:.2f}%)")

        # Count entries with multiple topics
        num_multi_topic = df["topic_clean"].apply(lambda x: len(x) > 1).sum()
        print(f"Share of entries with multiple topics: {num_multi_topic/total_entries}")

        # Count topic combinations (only for entries with 2 or more topics)
        combo_counter = Counter()
        for topics in df["topic_clean"]:
            if len(topics) > 1:
                sorted_combo = tuple(sorted(topics))
                combo_counter[sorted_combo] += 1

        print("Most frequent topic combinations:")
        for combo, count in combo_counter.most_common(10):
            print(f"{combo}: {count}x")


def main():
    
    dataframes = {}
    for filename in files:
        path = os.path.join(input_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["topic_clean"] = df["gpt_topic"].apply(extract_topics)
            dataframes[filename] = df
            print(f"Loaded {len(df)} rows from {path}")
        else:
            print(f"Not found: {path}")

    # read manually labeled data and add to df
    labeled_data = read_labeled_data(input_labeled_dir, files)

    for filename, df_labeled in labeled_data.items():
        df = dataframes[filename]
        print(f"Länge von df: {len(df)}")
        df_labeled = df_labeled.copy()
        print(f"Länge von df_labeled: {len(df_labeled)}")
        print(f"Anzahl der Topics in df_labeled: {df_labeled['topic_clean'].apply(len).sum()}")
        print(df_labeled['topic_clean'].value_counts())
        print(f"Datentypen: df: {df['id'].dtype}, df_labeled: {df_labeled['id'].dtype}")

        df = pd.merge(
            df,
            df_labeled[["id", "topic_clean"]],
            on="id",
            how="left",
            suffixes=("", "_labeled")
        )

        correction_counter = 0

        correction_counter = 0

        for idx, row in df.iterrows():
            if row["topic_clean"] == []:
                df.at[idx, "topic_clean"] = row["topic_clean_labeled"]
                correction_counter += 1

        print(f"Replaced {correction_counter} topics in {filename} with labeled data.")

        df.drop(columns=["topic_clean_labeled"], inplace=True)
        dataframes[filename] = df

    # merge with whole dataset
    merged_dataframes = merge_original_data(dataframes, preprocessed_dir)
    print(f"Merged {len(merged_dataframes)} dataframes with video information.")
    #saved merged dfs
    for filename, df_merged in merged_dataframes.items():
        output_path = os.path.join("results", "topic_analysis", "merged", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_merged.to_csv(output_path, index=False)
        print(f"Gespeichert: {output_path}")

    analyze_topic_combinations(dataframes)

    # remove rows without topics and save them
    dataframes_cleaned = save_rows_without_topic(dataframes, clean_dir)

    # save and analyse all topics
    process_and_save_files(dataframes_cleaned, clean_dir)
    count_topic_lengths(dataframes_cleaned)
    plot_topic_distribution(dataframes_cleaned, plot_dir)

    # analysis without topic "Wahlkampf"
    dataframes_no_wahlkampf = {
        filename: df.copy()
        for filename, df in dataframes_cleaned.items()
    }
    #save data without Wahlkampf
    cleaned_wahlkampf_dir = os.path.join(input_dir, "no_wahlkampf")
    cleaned_wahlkampf_dir_plots = os.path.join(plot_dir, "no_wahlkampf")
    os.makedirs(cleaned_wahlkampf_dir, exist_ok=True)

    dataframes_no_wahlkampf = remove_wahlkampf(dataframes_cleaned, cleaned_wahlkampf_dir)

    # process data without Wahlkampf
    process_and_save_files(dataframes_no_wahlkampf, cleaned_wahlkampf_dir)
    print("Without topic Wahlkampf:")
    count_topic_lengths(dataframes_no_wahlkampf)
    plot_topic_distribution(dataframes_no_wahlkampf, cleaned_wahlkampf_dir_plots)

    print(f"Merged {len(merged_dataframes)} dataframes with video information.")

    # process whole dataset (merged with all video information): timeline of topics, engagement metrics
    plot_topic_timeline(merged_dataframes, freq="W")
    overall, by_party = calculate_engagement_metrics(merged_dataframes, "plots/topic_analysis")
    plot_engagement_overall(overall, "plots/topic_analysis")
    plot_engagement_by_party(by_party, "plots/topic_analysis")

if __name__ == "__main__":
    main()