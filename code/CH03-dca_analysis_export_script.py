
# COS Verbs DCA Analysis + Export Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact

# Load data
trans_df = pd.read_excel("result_v2.xlsx", sheet_name="transitive_count")
intrans_df = pd.read_excel("result_v2.xlsx", sheet_name="intransitive_count")
cos_verbs = [line.strip() for line in open("COS verb_list_279.txt", "r", encoding="utf-8") if line.strip()]

# Merge counts
merged_df = pd.merge(trans_df[['verb', 'count']], intrans_df[['verb', 'count']],
                     on='verb', how='outer', suffixes=('_trans', '_intrans')).fillna(0)

# Filter for COS verbs
merged_cos_df = merged_df[merged_df['verb'].isin(cos_verbs)].copy()

# DCA Analysis
dca_rows = []

total_caus_cos = merged_cos_df['count_trans'].sum()
total_nca_cos = merged_cos_df['count_intrans'].sum()

for _, row in merged_cos_df.iterrows():
    O11 = int(row['count_trans'])
    O12 = int(row['count_intrans'])
    O21 = total_caus_cos - O11
    O22 = total_nca_cos - O12

    if (O11 + O12) == 0:
        continue

    table = [[O11, O12], [O21, O22]]
    chi2, p_chi2, _, _ = chi2_contingency(table)
    _, p_fisher = fisher_exact(table)
    odds_ratio = (O11 * O22) / (O12 * O21 + 1e-10)
    log_odds = np.log(odds_ratio + 1e-10)

    def assign_label(log_odds, p):
        if p >= 0.05:
            return "not significant"
        elif log_odds >= 2:
            return "strong attractor"
        elif 1 <= log_odds < 2:
            return "weak attractor"
        elif -1 < log_odds < 1:
            return "neutral"
        elif -2 < log_odds <= -1:
            return "slight repeller"
        else:
            return "strong repeller"

    dca_rows.append({
        "Verb": row['verb'],
        "Freq_in_Causative": O11,
        "Freq_in_Intransitive": O12,
        "Log_Likelihood": chi2,
        "p_value_chi2": p_chi2,
        "Fisher_p_value": p_fisher,
        "Log_Odds_Ratio": log_odds,
        "Label": assign_label(log_odds, p_chi2)
    })

dca_df = pd.DataFrame(dca_rows)
dca_df.sort_values(by="Log_Likelihood", ascending=False, inplace=True)

# Save DCA CSV
dca_df.to_csv("dca_result_COS.csv", index=False)

# --- Plot ---

# Color map
label_color_map = {
    "strong attractor": "red",
    "weak attractor": "gold",
    "neutral": "grey",
    "slight repeller": "lightskyblue",
    "strong repeller": "blue"
}

# Filter out non-significant
plot_df = dca_df[dca_df['Label'] != 'not significant']

# Select top-N per label for labeling
label_top_n = 8
selected_labels = []
for label in plot_df['Label'].unique():
    selected = plot_df[plot_df['Label'] == label].nlargest(label_top_n, 'Log_Likelihood')
    selected_labels.append(selected)
selected_labels_df = pd.concat(selected_labels).drop_duplicates(subset=["Verb"])

# Plot
plt.figure(figsize=(9,7))
for label, group in plot_df.groupby('Label'):
    plt.scatter(group["Log_Odds_Ratio"], -np.log10(group["p_value_chi2"]),
                label=label, color=label_color_map.get(label, "black"), edgecolor='k', alpha=0.8)

# Reference lines
plt.axvline(0, color='orange', linestyle='--', linewidth=1)
plt.axhline(-np.log10(0.05), color='orange', linestyle='--', linewidth=1)

# Simple jitter
def jitter_single(value, noise=0.5):
    return value + np.random.uniform(-noise, noise)

for _, row in selected_labels_df.iterrows():
    plt.text(jitter_single(row["Log_Odds_Ratio"]),
             jitter_single(-np.log10(row["p_value_chi2"])),
             row["Verb"], fontsize=8, weight='bold')

plt.xlabel("Log Odds Ratio (Causative vs Intransitive)")
plt.ylabel("-log10(p-value)")
plt.title("COS Verbs DCA (Journal Style, Clean Labels)")
plt.legend(title="Label", loc="upper right")
plt.tight_layout()
plt.savefig("dca_cos_final_plot.png", dpi=300)
plt.close()
