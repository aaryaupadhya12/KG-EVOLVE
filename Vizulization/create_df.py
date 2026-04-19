import os
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Create folder
# -----------------------
os.makedirs("plots", exist_ok=True)

# -----------------------
# Data
# -----------------------
df_main = pd.DataFrame({
    "Model": ["KGE", "BERT", "Combined"],
    "Hits@1": [0.1247, 0.3632, 0.3993],
    "Hits@3": [0.2352, 0.5755, 0.6138],
    "Hits@10": [0.5120, 0.7998, 0.8282],
    "MRR": [0.2382, 0.5062, 0.5381]
})

# -----------------------
# TURN OFF DISPLAY (IMPORTANT)
# -----------------------
plt.ioff()

# -----------------------
# 1. Model Performance
# -----------------------
ax = df_main.set_index("Model")[["Hits@1", "Hits@3", "Hits@10", "MRR"]].plot(kind="bar")
plt.title("Model Performance Comparison")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/model_performance.png", dpi=300)
plt.close()

# -----------------------
# 2. Ranking Changes
# -----------------------
labels = ["Improved", "Unchanged", "Worsened"]
values = [685, 130, 99]

plt.bar(labels, values)
plt.title("Ranking Changes")
plt.tight_layout()
plt.savefig("plots/ranking_changes.png", dpi=300)
plt.close()

# -----------------------
# 3. Routing Metrics
# -----------------------
metrics = ["pass@1", "efficiency", "loss", "agreement"]
vals = [0.83, 0.95, 0.04, 0.91]

plt.barh(metrics, vals)
plt.title("Routing Metrics")
plt.tight_layout()
plt.savefig("plots/routing_metrics.png", dpi=300)
plt.close()

# -----------------------
# 4. Grounded vs Lucky
# -----------------------
plt.bar(["Grounded", "Lucky"], [0.934, 0.066])
plt.title("Grounded vs Lucky")
plt.tight_layout()
plt.savefig("plots/grounded_vs_lucky.png", dpi=300)
plt.close()

print("Saved all plots in /plots")
