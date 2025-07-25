import os
import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from matplotlib.patches import Patch


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16
})

# configurations
input_dir = "results/topic_analysis/output_openai"
input_labeled_dir = "results/topic_analysis/cleaned/subset_labeled"
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
    """Maps short topic codes to full names."""
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
    """Reads labeled data from CSV files and processes the topics which have been added manually."""
    labeled_data = {}
    for filename in files:
        path = os.path.join(input_labeled_dir, filename)
        out_path = os.path.join(input_labeled_dir, "processed", filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep=None, engine="python", quoting=0)
                # get the clean topic by mapping short codes to full names
                df["topic_clean"] = df["gpt_topic"].apply(map_topics_short)
                df["topic_clean"] = df["topic_clean"].apply(lambda x: [x] if isinstance(x, str) else x)
                labeled_data[filename] = df
                df.to_csv(out_path, index=False)
            except Exception as e:
                print(f"Error reading labeled file {path}: {e}")
    return labeled_data

def extract_topics(text):
    """Extracts topics from LLM answers based on the keywords."""
    if pd.isna(text):
        return []
    found = set()
    # note: sometimes the LLM contains only partially the topic names, so some defined keywords are used to extract all topics correctly
    for match in re.findall(keyword_pattern, text):
        found.add(topic_keywords[match])
    return sorted(found)


def process_and_save_files(dataframes, clean_dir):
    """Processes and saves the dataframes to clean_dir."""
    try:
        for filename, df in dataframes.items():
            out_path = os.path.join(clean_dir, filename)
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error saving files: {e}")


def count_topic_lengths(dataframes):
    """Counts the number of unique topics in each row of the dataframes."""
    for filename, df in dataframes.items():
        df["topic_clean"] = df["topic_clean"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        topic_counts = df["topic_clean"].apply(len)
        count_summary = topic_counts.value_counts().sort_index()
        print(f"{filename}:")
        for n, count in count_summary.items():
            print(f"{n} unique topics: in {count} rows")


def save_rows_without_topic(dataframes, base_dir):
    """Saves rows without topics to a separate directory (for manual labeling)."""
    subset_dir = os.path.join(base_dir, "subset_not_labeled")
    os.makedirs(subset_dir, exist_ok=True)
    cleaned_dataframes = {}

    # checks whether topic_clean is empty
    for filename, df in dataframes.items():
        df["topic_clean"] = df["topic_clean"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_no_topic = df[df["topic_clean"].apply(lambda x: len(x) == 0)]
        df_with_topic = df[df["topic_clean"].apply(lambda x: len(x) > 0)]

        # save rows without topic
        if not df_no_topic.empty:
            out_path = os.path.join(subset_dir, filename)
            df_no_topic.to_csv(out_path, index=False)
            print(f"{len(df_no_topic)} rows without topic for {filename} saved in: {out_path}")

        print(f"Rows without topic: {len(df_no_topic)} in {filename}")
        cleaned_dataframes[filename] = df_with_topic

    return cleaned_dataframes


def plot_topic_distribution(dataframes, plot_dir):
    """Plots the distribution of topics for each party."""
    os.makedirs(plot_dir, exist_ok=True)

    topic_farben = {
        "Soziales & Arbeit": "#E69F00",     
        "Wirtschaft & Finanzen": "#97D4F7",    
        "Sicherheit & Ordnung": "#045280",    
        "Migration": "#F0E442",                 
        "Umwelt & Energie": "#009E73",     
        "Internationale Politik": "#A74800",  
        "Persönliches": "#CC79A7",         
        "Wahlkampf": "#999999"     
    }

    for filename, df in dataframes.items():
        # get topics
        df["topic_clean"] = df["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )
        df = df[df["topic_clean"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        if df.empty:
            print(f"No rows with topics in {filename}")
            continue

        # compute topic distribution
        all_topics = [topic for topics in df["topic_clean"] for topic in topics]
        topic_counts = pd.Series(all_topics).value_counts(normalize=True).sort_values(ascending=False) * 100

        # get topic colors
        colors = [topic_farben.get(topic, "#CCCCCC") for topic in topic_counts.index]

        # get party names and assign labels
        party_name = filename.replace(".csv", "")
        party_name_proc = {
            "CDU_CSU": "CDU/CSU",
            "Grüne": "Die Grünen",
            "Linke": "Die Linke"
        }.get(party_name, party_name)

        print(f"Topic distribution for {party_name_proc}:\n {topic_counts}")

        # Plot
        plt.figure(figsize=(8, 5))
        topic_counts.plot(
            kind="bar",
            color=colors,
            fontsize=16
        )

        plt.title(f"Anteile der Themenbereiche: {party_name_proc}", fontsize=16, fontweight="bold", pad=15)
        plt.ylabel("Anteil (%)", fontsize=16)
        plt.xlabel("Themenbereich", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.ylim(0, 55)
        plt.yticks(np.arange(0, 55, 10), fontsize=16)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{party_name}_topic_distribution_share.png"), bbox_inches="tight", dpi=300)
        plt.close()

def remove_wahlkampf(dataframes, output_dir):
    """For further investigation: remove Wahlkampf from topics because of its high frequency."""
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
    """Merges the topic data with the original video data."""
    merged_dataframes = {}

    for filename, df_topics in dataframes.items():
        # get party name
        partei = filename.replace(".csv", "")
        # get video data for the party
        video_path = os.path.join(preprocessed_dir, f"videos_{partei}.csv")

        if not os.path.exists(video_path):
            print(f"Video data not found for {partei}: {video_path}")
            continue

        try:
            df_videos = pd.read_csv(video_path)
        except Exception as e:
            print(f"Couldn't read {video_path}: {e}")
            continue

        # merge on these keys
        merge_keys = ["id", "username", "video_description", "voice_to_text"]

        df_merged = pd.merge(
            df_topics,
            df_videos,
            on=merge_keys,
            how="left" # keep all entries from topics
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
    """Plots the timeline of topics for each party."""

    os.makedirs("plots/topic_analysis", exist_ok=True)

    all_topics = [
        "Internationale Politik",
        "Migration",
        "Persönliches",
        "Sicherheit & Ordnung",
        "Soziales & Arbeit",
        "Umwelt & Energie",
        "Wahlkampf",
        "Wirtschaft & Finanzen"
    ]

    topic_colors = {
        "Soziales & Arbeit": "#E69F00",     
        "Wirtschaft & Finanzen": "#97D4F7",    
        "Sicherheit & Ordnung": "#045280",    
        "Migration": "#F0E442",                 
        "Umwelt & Energie": "#009E73",     
        "Internationale Politik": "#A74800",  
        "Persönliches": "#CC79A7",         
        "Wahlkampf": "#999999"     
    }
    # sort by name of topics
    all_topics = sorted(topic_colors.keys())

    # plot topic timelines
    for filename, df in merged_dataframes.items():
        partei = filename.replace(".csv", "")
        df = df.copy()

        df["create_time"] = pd.to_datetime(df["create_time"], errors="coerce")
        df = df.dropna(subset=["create_time", "topic_clean"])

        df["topic_clean"] = df["topic_clean"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        # get one row for each topic
        df = df.explode("topic_clean")
        # create period column for grouping (weekly or monthly, depending on freq)
        df["period"] = df["create_time"].dt.to_period(freq).dt.to_timestamp()

        grouped = df.groupby(["period", "topic_clean"]).size().unstack(fill_value=0)
        grouped = grouped.reindex(columns=all_topics, fill_value=0)
        grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
        grouped = grouped.loc[:pd.Timestamp("2025-02-23")] # stop at date of election

        # Format party names
        party_name_proc = {
            "CDU_CSU": "CDU/CSU",
            "Grüne": "Die Grünen",
            "Linke": "Die Linke"
        }.get(partei, partei)

        # plot
        fig, ax = plt.subplots(figsize=(12, 6))
        grouped.plot(
            ax=ax,
            color=[topic_colors[col] for col in grouped.columns],
            linewidth=2
        )
        # create legend
        ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Themenbereich",
        title_fontsize=16,
        fontsize=16
        )

        ax.set_title(f"Zeitlicher Verlauf der Themenanteile – {party_name_proc}", fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("Datum", fontsize=18)
        ax.set_ylabel("Anteil an geposteten Videos (%)", fontsize=18)

        ax.set_ylim(0, 70)
        ax.set_yticks(np.arange(0, 71, 10))

        ax.set_xticks(grouped.index)
        # create x labels with date format
        ax.set_xticklabels([d.strftime("%d.%m.%y") for d in grouped.index], rotation=45, ha="right", fontsize=18)
        ax.tick_params(axis="y", labelsize=18)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"plots/topic_analysis/topic_timeline_{partei}.png", bbox_inches="tight", dpi=300)
    
        plt.close()
        print(f"Saved timeline for {partei}")

def plot_topic_timeline_stacked(merged_dataframes, freq="W"):
    """Plots the stacked timeline of topics for each party. Similar to timeline plot above but stacked area chart."""

    os.makedirs("plots/topic_analysis", exist_ok=True)

    all_topics = [
        "Internationale Politik",
        "Migration",
        "Persönliches",
        "Sicherheit & Ordnung",
        "Soziales & Arbeit",
        "Umwelt & Energie",
        "Wirtschaft & Finanzen",
        "Wahlkampf"
    ]

    topic_colors = {
        "Soziales & Arbeit": "#E69F00",     
        "Wirtschaft & Finanzen": "#97D4F7",    
        "Sicherheit & Ordnung": "#045280",    
        "Migration": "#F0E442",                 
        "Umwelt & Energie": "#009E73",     
        "Internationale Politik": "#A74800",  
        "Persönliches": "#CC79A7",         
        "Wahlkampf": "#999999"     
    }

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
        grouped = grouped.reindex(columns=all_topics, fill_value=0)
        grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
        grouped = grouped.loc[:pd.Timestamp("2025-02-23")]

        party_name_proc = {
            "CDU_CSU": "CDU/CSU",
            "Grüne": "Die Grünen",
            "Linke": "Die Linke"
        }.get(partei, partei)

        fig, ax = plt.subplots(figsize=(12, 6))
        grouped.plot.area(
            ax=ax,
            color=[topic_colors[col] for col in grouped.columns]
        )

        ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Themenbereich",
        title_fontsize=16,
        fontsize=16
        )

        ax.set_title(f"Zeitlicher Verlauf der Themenanteile – {party_name_proc}", fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("Datum", fontsize=18)
        ax.set_ylabel("Anteil an geposteten Videos (%)", fontsize=18)

        ax.set_xticks(grouped.index)
        ax.set_xticklabels([d.strftime("%d.%m.%y") for d in grouped.index], rotation=45, ha="right", fontsize=18)
        ax.tick_params(axis="y", labelsize=18)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"plots/topic_analysis/stacked_topic_timeline_{partei}.png", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved timeline for {partei}")

def calculate_engagement_metrics(merged_dataframes, output_dir):
    """Calculates engagement metrics for each topic and party."""
    all_data = []

    for filename, df in merged_dataframes.items():
        partei = filename.replace(".csv", "")
        df = df.copy()

        # preparation of topic_clean
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

    # get the metrics for each topic
    overall = combined_df.groupby("topic_clean")[metrics].mean().round(2)
    # get the metrics for each topic by party
    by_party = combined_df.groupby(["partei", "topic_clean"])[metrics].mean().round(2)

    # Save as csv
    os.makedirs(output_dir, exist_ok=True)
    overall.to_csv(os.path.join(output_dir, "engagement_by_topic.csv")) # by topic
    by_party.to_csv(os.path.join(output_dir, "engagement_by_party_topic.csv")) # by party

    print("Engagement-metrics compete:")
    print(overall)

    print("Engagement-Metriken by party:")
    print(by_party)

    return overall, by_party

def plot_engagement_overall(overall_df, output_dir):
    """Plots the overall engagement metrics by topic."""
    os.makedirs(output_dir, exist_ok=True)

    ax = overall_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Durchschnittliche Engagement-Metriken pro Topic (gesamt)", fontweight="bold", pad=15)
    plt.ylabel("Durchschnitt pro Video")
    plt.xlabel("Topic")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engagement_by_topic.png"), dpi=300)
    plt.close()
    print("Saved: engagement_by_topic.png")

def plot_engagement_by_party(by_party_df, output_dir):
    """Plots the engagement metrics by party and topic."""
    os.makedirs(output_dir, exist_ok=True)

    # colors and labels as before
    parteien = ["AfD", "CDU_CSU", "Grüne", "Linke", "SPD"]
    topic_colors = {
        "Soziales & Arbeit": "#E69F00",     
        "Wirtschaft & Finanzen": "#97D4F7",    
        "Sicherheit & Ordnung": "#045280",    
        "Migration": "#F0E442",                 
        "Umwelt & Energie": "#009E73",     
        "Internationale Politik": "#A74800",  
        "Persönliches": "#CC79A7",         
        "Wahlkampf": "#999999"     
    }
    legenden_labels = {
        "AfD": "AfD",
        "CDU_CSU": "CDU/CSU",
        "Grüne": "Die Grünen",
        "Linke": "Die Linke",
        "SPD": "SPD"
    }

    # Names for the metrics
    metric_labels = {
        "like_count": "Anzahl an Likes",
        "view_count": "Anzahl an Ansichten",
        "share_count": "Anzahl an Weiterleitungen",
        "comment_count": "Anzahl an Kommentaren"
    }

    for metric in by_party_df.columns:

        # create pivot table
        pivot = by_party_df.reset_index().pivot(index="partei", columns="topic_clean", values=metric)
        # get in right order
        pivot = pivot.loc[parteien]
        pivot.index = [legenden_labels.get(p, p) for p in pivot.index]

        ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        color=[topic_colors.get(t, "#CCCCCC") for t in pivot.columns]
        )

        pivot.to_csv(os.path.join("results/topic_analysis/metrics", f"{metric}_by_party.csv"))

        ax.legend(
            title="Themenbereich",
            title_fontsize=16,
            fontsize=16,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5)
        )

        plt.title(f"Durchschnittliche {metric_labels.get(metric, metric)} pro Themenbereich & Partei", fontsize=18, fontweight="bold", pad=15)
        plt.ylabel(metric_labels.get(metric, metric), fontsize=18)
        plt.xlabel("Themenbereich", fontsize=18)
        plt.xticks(rotation=45, ha="right", fontsize=18)
        plt.yticks(fontsize=18)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_by_party.png"), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved plot: {metric}_by_party.png")

def analyze_topic_combinations(dataframes):
    """Analyzes topic combinations in the dataframes."""
    for filename, df in dataframes.items():
        print(f"Analysis for {filename}:")

        # preparation of topic_clean
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

        print("Topic combinations and their counts:")
        for combo, count in combo_counter.most_common(51):
            print(f"{combo}: {count}x")

def main():
    
    dataframes = {}
    # get topic data from files
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

    # add manually labeled topics to dataframes
    for filename, df_labeled in labeled_data.items():
        df = dataframes[filename]
        print(f"Length of df: {len(df)}")
        # check results
        print(f"Length of df_labeled: {len(df_labeled)}")
        print(f"Number of topics in df_labeled: {df_labeled['topic_clean'].apply(len).sum()}")
        print(df_labeled['topic_clean'].value_counts())
        print(f"Data types: df: {df['id'].dtype}, df_labeled: {df_labeled['id'].dtype}")

        # merge labeled topics into the main dataframe
        df = pd.merge(
            df,
            df_labeled[["id", "topic_clean"]],
            on="id",
            how="left",
            suffixes=("", "_labeled")
        )

        # counter for checks later
        correction_counter = 0

        # use labeled topics if original topics are empty
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
    # saved merged dfs
    for filename, df_merged in merged_dataframes.items():
        output_path = os.path.join("results", "topic_analysis", "merged", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_merged.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    # get topic combinations
    analyze_topic_combinations(dataframes)

    # remove rows without topics and save them 
    # (note: this is for manual labeling, in this version the manual labeling is already done and merged into the data above, before the manual labeling this step was necessary to get the data without topics)
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
    # save data without Wahlkampf
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
    plot_topic_timeline_stacked(merged_dataframes, freq="W")
    overall, by_party = calculate_engagement_metrics(merged_dataframes, "plots/topic_analysis")
    plot_engagement_overall(overall, "plots/topic_analysis")
    plot_engagement_by_party(by_party, "plots/topic_analysis")

if __name__ == "__main__":
    main()