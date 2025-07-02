#!/usr/bin/env python3
"""
analysis.py ─ End-to-end analytics for the synthetic sports-equipment survey
---------------------------------------------------------------------------
* Classification  : Predict willingness to try the new delivery app.
* Clustering      : Segment users with K-Means (default k=4).
* Regression      : Predict monthly equipment budget.
* Assoc.-Rules    : Discover preference/challenge/itemset relationships.

Outputs (saved to ./outputs):
  ├─ classification_report.txt
  ├─ cluster_profiles.csv
  ├─ cluster_sizes.png
  ├─ regression_metrics.txt
  ├─ frequent_itemsets.csv
  └─ association_rules.csv
"""

import argparse, os, json, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    r2_score, mean_squared_error, silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------------------------------------------------------- #
# 1 ── Argument parsing
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Analytics for sports-app survey")
parser.add_argument("--csv", required=True, help="Path to survey CSV")
parser.add_argument("--k", type=int, default=4, help="K for K-Means (default=4)")
args = parser.parse_args()

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2 ── Load dataset
# --------------------------------------------------------------------------- #
try:
    df = pd.read_csv(args.csv)
except FileNotFoundError:
    sys.exit(f"❌  File not found: {args.csv}")

print("🔹 Dataset loaded:", df.shape)

# --------------------------------------------------------------------------- #
# 3 ── Expand list-like columns stored as strings
# --------------------------------------------------------------------------- #
multiselect_cols = [
    "Sports_Played", "Equipment_Owned",
    "Purchase_Influencers", "Purchase_Challenges"
]

def parse_list(x):
    """Convert string representation of list to actual list."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "[]":
        return []
    # Replace single quotes with double quotes so json.loads works.
    return json.loads(x.replace("'", '"'))

for col in multiselect_cols:
    if col in df.columns:
        df[col] = df[col].apply(parse_list)
    else:
        print(f"⚠️  Column missing in CSV: {col}")

# --------------------------------------------------------------------------- #
# 4 ── Classification : Willingness_To_Use_App
# --------------------------------------------------------------------------- #
target_col = "Willingness_To_Use_App"
if target_col not in df.columns:
    sys.exit(f"❌  Target column `{target_col}` not found in CSV.")

target_order = ["Very Unlikely", "Unlikely", "Neutral", "Likely", "Very Likely"]
y = df[target_col].map({v: i for i, v in enumerate(target_order)})

X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
cat_cols = X.select_dtypes(include="object").columns.to_list()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

clf = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=300, multi_class="auto"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🏷️  Classification accuracy: {acc:.3f}")

report_txt = classification_report(
    y_test, y_pred, target_names=target_order, digits=3
)
with open(os.path.join(OUTDIR, "classification_report.txt"), "w") as f:
    f.write(report_txt)
print("   ↳ Saved classification_report.txt")

# --------------------------------------------------------------------------- #
# 5 ── Clustering : K-Means segments
# --------------------------------------------------------------------------- #
cluster_pipe = Pipeline([
    ("prep", preprocess),
    ("model", KMeans(n_clusters=args.k, random_state=42, n_init="auto"))
])

X_cluster = cluster_pipe["prep"].fit_transform(X)
labels = cluster_pipe["model"].fit_predict(X_cluster)
sil = silhouette_score(X_cluster, labels)
print(f"\n📊 K-Means k={args.k} silhouette score: {sil:.3f}")

df["Cluster"] = labels

profile_cols = ["Age", "Monthly_Income", "Monthly_Equipment_Budget"]
(df.groupby("Cluster")[profile_cols]
     .mean()
     .round(1)
     .to_csv(os.path.join(OUTDIR, "cluster_profiles.csv"))
)

plt.figure()
pd.Series(labels).value_counts().sort_index().plot(kind="bar")
plt.title("Cluster Sizes")
plt.ylabel("Count")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "cluster_sizes.png"))
plt.close()
print("   ↳ Cluster outputs saved")

# --------------------------------------------------------------------------- #
# 6 ── Regression : Monthly_Equipment_Budget
# --------------------------------------------------------------------------- #
budget_col = "Monthly_Equipment_Budget"
reg_y = df[budget_col]
reg_X = df.drop(columns=[budget_col, "Cluster", target_col])

num_cols_reg = reg_X.select_dtypes(include=["int64", "float64"]).columns.to_list()
cat_cols_reg = reg_X.select_dtypes(include="object").columns.to_list()

reg_pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols_reg),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_reg)
])

reg_pipe = Pipeline([
    ("prep", reg_pre),
    ("model", LinearRegression())
])

reg_pipe.fit(reg_X, reg_y)
reg_pred = reg_pipe.predict(reg_X)
r2 = r2_score(reg_y, reg_pred)
rmse = mean_squared_error(reg_y, reg_pred, squared=False)
with open(os.path.join(OUTDIR, "regression_metrics.txt"), "w") as f:
    f.write(f"R2  : {r2:.3f}\nRMSE: {rmse:.1f}\n")
print(f"\n💰 Regression ⇒ R²={r2:.3f} | RMSE={rmse:.1f}")

# --------------------------------------------------------------------------- #
# 7 ── Association-Rule Mining
# --------------------------------------------------------------------------- #
# Build a binary “basket” matrix from multi-select columns
basket = pd.DataFrame(index=df.index)
for col in multiselect_cols:
    one_hot = pd.get_dummies(
        df[col].explode()
    ).groupby(level=0).max()  # bring back to original rows
    basket = basket.reindex(one_hot.index.union(basket.index)).fillna(0)
    basket = basket.join(one_hot, how="left").fillna(0)

basket = basket.astype(int)
freq_items = apriori(basket, min_support=0.1, use_colnames=True)
rules = (association_rules(freq_items, metric="lift", min_threshold=1.2)
         .sort_values("confidence", ascending=False))

freq_items.to_csv(os.path.join(OUTDIR, "frequent_itemsets.csv"), index=False)
rules.to_csv(os.path.join(OUTDIR, "association_rules.csv"), index=False)
print(f"\n🔗 Generated {len(freq_items)} frequent itemsets "
      f"and {len(rules)} rules (saved).")

# --------------------------------------------------------------------------- #
print(f"\n✅ All done!  See '{OUTDIR}/' for results.")
