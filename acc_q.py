import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Configuration
api_key = os.getenv('OPENAI_API_KEY', 'your keys')
client = OpenAI(api_key=api_key)
RESULT_DIR = "./logs"
# GT_DIR = "./GT_LLMSTPA"
GT_DIR = "./GT_CAST"
# COS_CSV = "cosine_similarity_report.csv"
COS_CSV = "cast_cosine_similarity_report.csv"
# COMBINED_CSV = "hybrid_accuracy_report.csv"
COMBINED_CSV = "cast_hybrid_accuracy_report.csv"
LOW_SIM_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.6
# GPT_CSV = "comparison_report.csv"  # must exist from previous run
GPT_CSV = "cast_comparison_report.csv"  # must exist from previous run

# Embedding Function
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return response.data[0].embedding

# Read Result JSON fields
def read_result_text(path):
    with open(path, 'r') as f:
        data = json.load(f)
    parts = [data.get(k, '') for k in ["scene_description", "unsafe_control_actions", "loss_scenario_log", "safe_scenario_log"]]
    return "\n\n".join(parts)

# Collect all GT txt files
def index_ground_truth():
    gt_map = {}
    for root, _, files in os.walk(GT_DIR):
        for file in files:
            if file.endswith(".txt"):
                base = os.path.splitext(file)[0]
                gt_map[base] = os.path.join(root, file)
    return gt_map

# Main
if __name__ == "__main__":
    gt_files = index_ground_truth()
    results = []
    low_sim_cases = []

    for file in os.listdir(RESULT_DIR):
        if not file.endswith("_log.json"):
            continue

        base = file.replace("_log.json", "")
        result_path = os.path.join(RESULT_DIR, file)
        gt_path = gt_files.get(base)

        if not gt_path or not os.path.exists(gt_path):
            print(f"Skipping unmatched: {base}")
            continue

        with open(gt_path, 'r') as f:
            gt_text = f.read()
        result_text = read_result_text(result_path)

        try:
            emb_gt = get_embedding(gt_text)
            emb_result = get_embedding(result_text)
            cos_sim = cosine_similarity([emb_gt], [emb_result])[0][0]
        except Exception as e:
            print(f"Embedding failed for {base}: {e}")
            cos_sim = -1

        classification = "Accurate" if cos_sim >= SIMILARITY_THRESHOLD else "Inaccurate"
        results.append({"file": base, "cosine_similarity": cos_sim, "classification": classification})

        if cos_sim < LOW_SIM_THRESHOLD:
            low_sim_cases.append({"file": base, "cosine_similarity": cos_sim})

    df_cos = pd.DataFrame(results)
    df_cos.to_csv(COS_CSV, index=False)
    print(f"Saved cosine similarity report to {COS_CSV}")

    # Merge with GPT CSV if available
    if os.path.exists(GPT_CSV):
        df_gpt = pd.read_csv(GPT_CSV)
        df_combined = pd.merge(df_cos, df_gpt, on="file", how="inner")
        df_combined.to_csv(COMBINED_CSV, index=False)
        print(f"Saved hybrid GPT + Cosine accuracy report to {COMBINED_CSV}")
    else:
        df_combined = df_cos.copy()

    # Save low similarity cases
    if low_sim_cases:
        low_sim_df = pd.DataFrame(low_sim_cases)
        low_sim_df.to_csv("low_similarity_cases.csv", index=False)
        print("Exported low similarity cases < 0.5 to 'low_similarity_cases.csv'")

    # Plot Cosine Similarity Chart
    plt.figure(figsize=(14, 6))
    df_sorted = df_cos.sort_values(by="cosine_similarity", ascending=False)
    plt.bar(df_sorted["file"], df_sorted["cosine_similarity"], color=["green" if x >= SIMILARITY_THRESHOLD else "red" for x in df_sorted["cosine_similarity"]])
    plt.axhline(SIMILARITY_THRESHOLD, color='blue', linestyle='--', label=f"Threshold = {SIMILARITY_THRESHOLD}")
    plt.xticks(rotation=90)
    plt.xlabel("Sample Filename")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity per Sample (GT vs Result)")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("cosine_similarity_chart.png")
    plt.savefig("cast_cosine_similarity_chart.png")
    plt.show()

    # Pie chart for classification
    plt.figure(figsize=(6, 6))
    class_counts = df_cos["classification"].value_counts()
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=["green", "red"])
    plt.title("Accuracy Classification by Cosine Similarity")
    plt.tight_layout()
    # plt.savefig("cosine_accuracy_pie.png")
    plt.savefig("cast_cosine_accuracy_pie.png")
    plt.show()

    # Scatter plot: cosine vs GPT label
    if "similarity" in df_combined.columns:
        label_map = {
            "Exact Match": 0,
            "Minor Differences": 1,
            "Major Differences": 2,
            "Mismatch": 3
        }
        df_combined["gpt_numeric"] = df_combined["similarity"].map(label_map)

        plt.figure(figsize=(8, 6))
        plt.scatter(df_combined["cosine_similarity"], df_combined["gpt_numeric"], c=df_combined["gpt_numeric"], cmap='viridis', alpha=0.7)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("GPT Similarity Label (0=Exact, 3=Mismatch)")
        plt.title("Scatter Plot: Cosine Similarity vs GPT Label")
        plt.yticks(ticks=list(label_map.values()), labels=list(label_map.keys()))
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig("cosine_vs_gpt_scatter.png")
        plt.savefig("cast_cosine_vs_gpt_scatter.png")
        plt.show()