import os
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Output folder
# -----------------------
os.makedirs("plots", exist_ok=True)
plt.ioff()  # disables GUI rendering

# -----------------------
# Main Results Data
# -----------------------
df_main = pd.DataFrame({
    "Model": ["KGE (RotatE)", "BERT", "Combined"],
    "Hits@1": [0.0091, 0.0937, 0.0665],
    "Hits@3": [0.0544, 0.1752, 0.1752],
    "Hits@10": [0.3595, 0.3897, 0.4411],
    "MRR": [0.1074, 0.1917, 0.1798]
})

# -----------------------
# Rank change
# -----------------------
df_rank = pd.DataFrame({
    "Metric": ["Improved", "Unchanged", "Worsened"],
    "Value": [169, 8, 154]
})

# -----------------------
# Routing metrics
# -----------------------
df_route = pd.DataFrame({
    "Metric": [
        "pass@1_routed",
        "pass@2_total",
        "pass@2_grounded",
        "pass@2_lucky",
        "routing_efficiency",
        "routing_loss",
        "A_lucky_rate",
        "B_lucky_rate",
        "agreement_rate"
    ],
    "Value": [
        0.9059,
        0.9412,
        0.3059,
        0.6353,
        0.9625,
        0.0353,
        0.7843,
        0.6623,
        0.6235
    ]
})

# -----------------------
# 1. Model comparison
# -----------------------
ax = df_main.set_index("Model")[["Hits@1", "Hits@3", "Hits@10", "MRR"]].plot(kind="bar")
plt.title("Model Performance Comparison")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/model_performance.png", dpi=300)
plt.close()

# -----------------------
# 2. Rank changes
# -----------------------
plt.bar(df_rank["Metric"], df_rank["Value"])
plt.title("Rank Changes vs RotatE")
plt.tight_layout()
plt.savefig("plots/rank_changes.png", dpi=300)
plt.close()

# -----------------------
# 3. Routing metrics
# -----------------------
plt.figure(figsize=(8, 5))
plt.barh(df_route["Metric"], df_route["Value"])
plt.title("Routing Performance Metrics")
plt.tight_layout()
plt.savefig("plots/routing_metrics.png", dpi=300)
plt.close()

# -----------------------
# DONE
# -----------------------
print("All plots saved inside /plots folder")