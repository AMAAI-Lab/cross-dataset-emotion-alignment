import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA

# Load the data
npz_file_path = "label_expanded_jamendo.npz"
data = np.load(npz_file_path, allow_pickle=True)
labels = data["labels"]
text_features = data["text_features"]
npz_file_path = "label_expanded_CAL500.npz"
data = np.load(npz_file_path, allow_pickle=True)
labels_cal = data["labels"]
print(f"labels_cal: {len(labels_cal)}")
text_features_cal = data["text_features"]
npz_file_path = "label_expanded_emotify.npz"
data = np.load(npz_file_path, allow_pickle=True)
labels_emotify = data["labels"]
print(f"labels_cal: {len(labels_emotify)}")
text_features_emotify = data["text_features"]
text_features = np.concatenate((text_features, text_features_cal, text_features_emotify), axis=0)
labels = np.concatenate((labels, labels_cal, labels_emotify), axis=0)
print(text_features.shape)
# Assuming labels are in a 2D array format, if not, reshape or preprocess accordingly
# For example, if labels are 1D, you might need to reshape or use PCA to reduce dimensions

# Apply PCA to reduce dimensions to 2 for visualization purposes
pca = PCA(n_components=2)
labels_2d = pca.fit_transform(text_features)

# Estimate bandwidth for Mean Shift
bandwidth = estimate_bandwidth(
    text_features, quantile=0.03, n_samples=82
)  # for all three combined
# bandwidth = estimate_bandwidth(text_features, quantile=0.03, n_samples=73) #for jamendo and CAL500 combined
# bandwidth = estimate_bandwidth(text_features, quantile=0.025, n_samples=65) #for jasmendo and emotify combined
# bandwidth = estimate_bandwidth(text_features, quantile=0.04, n_samples=26) #for cal500 and emotify combined
# bandwidth = 0.000000001
print(f"Estimated bandwidth: {bandwidth}")

# Apply Mean Shift clustering
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(text_features)
cluster_labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_

unique_clusters = np.unique(cluster_labels)
print(f"Unique number of clusters: {len(unique_clusters)}")

# Get the cluster center coordinates for each label
cluster_center_coords = cluster_centers[cluster_labels]
# print(cluster_centers.shape)

labels_by_cluster = {cluster: [] for cluster in unique_clusters}
for label, cluster_label in zip(labels, cluster_labels):
    labels_by_cluster[cluster_label].append(label)

# print("Labels by Cluster Group:")
print("Labels by Cluster Group:")
for cluster, cluster_labels_list in labels_by_cluster.items():
    print(f"Cluster {cluster}: {cluster_labels_list}")

cluster_labels = np.array(cluster_labels)

plt.figure(figsize=(10, 6))
plt.scatter(
    labels_2d[:, 0], labels_2d[:, 1], c=cluster_labels.astype(float), cmap="viridis", marker="o"
)
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1], s=300, c="red", marker="x"
)  # Plot cluster centers
plt.title("Mean Shift Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")

# Save the plot to a file
plot_file_path = "mean_shift_clustering.png"
plt.savefig(plot_file_path)
plt.close()
# Save the cluster assignments and cluster center coordinates to a JSON file
cluster_data = []
for label, cluster_label, center_coord in zip(labels, cluster_labels, cluster_center_coords):
    cluster_data.append(
        {
            "Label": label,
            "Cluster": int(cluster_label),
            "Cluster Center": center_coord.tolist(),  # Convert numpy array to list for JSON serialization
        }
    )

json_file_path = "cluster_assignments_all.json"
with open(json_file_path, "w") as json_file:
    json.dump(cluster_data, json_file, indent=4)

print(f"Plot saved to {plot_file_path}")
print(f"Cluster assignments saved to {json_file_path}")
