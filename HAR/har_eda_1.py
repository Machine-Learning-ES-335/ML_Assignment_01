# har_eda.py task 1 eda
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATASET_PATH = r"C:\Users\hp\Downloads\HAR VSC\UCI HAR Dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
INERTIAL_FOLDER = os.path.join(TRAIN_PATH, "Inertial Signals")
OUT_DIR = os.path.join(os.path.dirname(DATASET_PATH), "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
SAMPLE_RATE = 50  # Hz


# load data

def load_inertial(axis_name):
    fname = os.path.join(INERTIAL_FOLDER, f"total_acc_{axis_name}_train.txt")
    return pd.read_csv(fname, sep=r'\s+', header=None)

print("Loading inertial signals...")
acc_x = load_inertial("x")
acc_y = load_inertial("y")
acc_z = load_inertial("z")
print(f"Shapes: x={acc_x.shape}, y={acc_y.shape}, z={acc_z.shape}")

y_train = pd.read_csv(os.path.join(TRAIN_PATH, "y_train.txt"), header=None, names=["activity"])
activity_labels = pd.read_csv(os.path.join(DATASET_PATH, "activity_labels.txt"),
                              sep=r'\s+', header=None, index_col=0)[1].to_dict()

# compute L2 norm (linear/total acceleration)
acc_x_np = acc_x.values.astype(float)
acc_y_np = acc_y.values.astype(float)
acc_z_np = acc_z.values.astype(float)
total_acc = np.sqrt(acc_x_np**2 + acc_y_np**2 + acc_z_np**2)  # shape (n_samples, n_timesteps)


# 1. Plot one sample waveform per activity (6 panels)

unique_acts = sorted(y_train['activity'].unique())
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for i, act in enumerate(unique_acts):
    idx = np.where(y_train['activity'].values == act)[0][0]
    ax = axes.flat[i]
    ax.plot(total_acc[idx, :])
    ax.set_title(f"{act}: {activity_labels[act]}")
    ax.set_xlabel("time samples")
    ax.set_ylabel("acc (L2)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "waveforms_one_per_activity.png"), dpi=150)
plt.show()


# 2. Static vs Dynamic: compare summary stats of L2 per activity

df_stats = pd.DataFrame({
    "activity": y_train['activity'].values,
    "activity_name": y_train['activity'].map(activity_labels),
    "mean_l2": total_acc.mean(axis=1),
    "std_l2": total_acc.std(axis=1),
    "max_l2": total_acc.max(axis=1)
})

plt.figure(figsize=(10,5))
sns.boxplot(x="activity_name", y="mean_l2", data=df_stats, order=[activity_labels[a] for a in unique_acts])
plt.xticks(rotation=45)
plt.title("Distribution of mean total acceleration by activity")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_mean_l2_by_activity.png"), dpi=150)
plt.show()

# Quick numeric summary
print(df_stats.groupby("activity_name")[["mean_l2", "std_l2"]].agg(['mean', 'std']).round(4))


# 3.A PCA on total_acc timeseries -> 2 components
# Each sample (row) is the timeseries vector -> PCA reduces to 2 numbers per sample

scaler = StandardScaler()
total_acc_scaled = scaler.fit_transform(total_acc)  # shape (n_samples, n_timesteps)
pca = PCA(n_components=2)
total_acc_pca = pca.fit_transform(total_acc_scaled)

pca_df = pd.DataFrame({
    "PC1": total_acc_pca[:,0],
    "PC2": total_acc_pca[:,1],
    "activity": y_train["activity"].values,
    "activity_name": y_train["activity"].map(activity_labels)
})

plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="activity_name", s=30, alpha=0.8)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.title("PCA of total_acc timeseries (2 components)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_total_acc_timeseries.png"), dpi=150)
plt.show()

# 3.B TSFEL features (can be slow). Example extraction.
try:
    import tsfel
    print("Extracting TSFEL features (this may take time).")
    # get a default feature config  (may choose domains)
    cfg = tsfel.get_features_by_domain()
    tsfel_features_list = []

    for i in range(total_acc.shape[0]):
        # convert to pandas.Series or 1D array
        ts = pd.Series(total_acc[i, :])
        feats = tsfel.time_series_features_extractor(cfg, ts, fs=SAMPLE_RATE)
        # results a 1-row dataframe; append the row as a series
        tsfel_features_list.append(feats.iloc[0])
    tsfel_df = pd.DataFrame(tsfel_features_list).reset_index(drop=True)
    print("TSFEL features shape:", tsfel_df.shape)
    # PCA on TSFEL features
    tsfel_df = tsfel_df.fillna(0)  # simple NA handling
    scaler2 = StandardScaler()
    tsfel_scaled = scaler2.fit_transform(tsfel_df)
    pca2 = PCA(n_components=2)
    tsfel_pca = pca2.fit_transform(tsfel_scaled)
    pca_tsfel_df = pd.DataFrame({"PC1": tsfel_pca[:,0], "PC2": tsfel_pca[:,1],
                                 "activity": y_train["activity"].values, "activity_name": y_train["activity"].map(activity_labels)})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_tsfel_df, x="PC1", y="PC2", hue="activity_name", s=30, alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title("PCA of TSFEL features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_tsfel.png"), dpi=150)
    plt.show()
except Exception as e:
    print("TSFEL not run or failed. Install tsfel and retry (pip install tsfel). Error:", e)


# 3.C Dataset-provided features -> load X_train.txt and features.txt (these are the 561 features in original UCI)
try:
    X_train = pd.read_csv(os.path.join(TRAIN_PATH, "X_train.txt"), sep=r'\s+', header=None)
    feat_names = pd.read_csv(os.path.join(DATASET_PATH, "features.txt"), sep=r'\s+', header=None, index_col=0)[1].tolist()
    if X_train.shape[1] == len(feat_names):
        X_train.columns = feat_names
    else:
        print("Feature name count mismatch; continuing with numeric column names.")
    scaler3 = StandardScaler()
    X_scaled = scaler3.fit_transform(X_train)
    pca3 = PCA(n_components=2)
    X_pca = pca3.fit_transform(X_scaled)
    pca_dataset_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                                   "activity": y_train["activity"].values, "activity_name": y_train["activity"].map(activity_labels)})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pca_dataset_df, x="PC1", y="PC2", hue="activity_name", s=30, alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title("PCA of provided dataset features (X_train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_dataset_features.png"), dpi=150)
    plt.show()
except Exception as e:
    print("Could not load X_train.txt/features.txt. Error:", e)


# 4. Correlation matrix for TSFEL features and dataset features (if available)

def find_high_corr_pairs(df, thresh=0.95, top_n=20):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    # get pairs > thresh
    rows, cols = np.where(upper > thresh)
    for r, c in zip(rows, cols):
        pairs.append((df.columns[r], df.columns[c], corr.iloc[r, c]))
    # sort
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs_sorted[:top_n]

if 'tsfel_df' in globals():
    print("Top highly correlated TSFEL feature pairs:")
    for a,b,val in find_high_corr_pairs(tsfel_df, thresh=0.95, top_n=20):
        print(f"{a} <-> {b} : {val:.3f}")

if 'X_train' in globals():
    print("Top highly correlated dataset feature pairs (from X_train):")
    for a,b,val in find_high_corr_pairs(X_train, thresh=0.95, top_n=20):
        print(f"{a} <-> {b} : {val:.3f}")

print("EDA script finished. Figures saved to:", OUT_DIR)
