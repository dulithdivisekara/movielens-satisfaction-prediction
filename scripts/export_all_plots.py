"""
Export All Figures to results/eda_visualizations/ for Presentation Slides
Group 08 - MovieLens 32M
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs('results/eda_visualizations', exist_ok=True)
os.makedirs('results/outputs', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Exporting high-resolution visual assets to results/eda_visualizations/...")

# 1. Movie catalog and ratings EDA
movies_df = pd.read_csv('data/processed/benchmark_movies.csv')
ratings_df = pd.read_csv('data/processed/benchmark_ratings.csv')
movies_feat = pd.read_csv('data/processed/movie_features.csv')

# Genre distribution
plt.figure(figsize=(10, 6))
genre_series = movies_df['genres'].str.split('|').explode().value_counts()
sns.barplot(x=genre_series.values, y=genre_series.index, palette='crest')
plt.title("Movie Catalog Distribution Across Genres", fontsize=14, fontweight='bold')
plt.xlabel("Number of Movies")
plt.tight_layout()
plt.savefig('results/eda_visualizations/01_genre_distribution.png', dpi=300)
plt.close()

# Rating distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=ratings_df, x='rating', palette='viridis', edgecolor='black')
plt.title("Rating Frequency Distribution (0.5 to 5.0 Stars)", fontsize=13, fontweight='bold')
plt.xlabel("Rating Value")
plt.ylabel("Rating Count")
plt.tight_layout()
plt.savefig('results/eda_visualizations/02_rating_distribution.png', dpi=300)
plt.close()

# 2. PCA Clustering Map
genre_cols = [c for c in movies_feat.columns if c.startswith('genre_')]
feat_cols = ['movie_mean_rating', 'movie_rating_count', 'movie_rating_std', 'release_year'] + genre_cols
X_sc = StandardScaler().fit_transform(movies_feat[feat_cols])
km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X_sc)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
centroids_pca = pca.transform(km.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='black', edgecolors='white', linewidth=2, label='Cluster Centroids')
plt.title("2D PCA Projection of Catalog Clusters", fontsize=14, fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('results/eda_visualizations/03_kmeans_pca_clusters.png', dpi=300)
plt.close()

# 3. Supervised Models Variance Error Bars
models = ['Random Forest\n(Member 3)', 'SVM (RBF Kernel)\n(Member 4)', 'Deep MLP Neural Net\n(Member 6)']
f1_scores = [0.7385, 0.7321, 0.7408]
f1_stds = [0.0038, 0.0042, 0.0040]
f1_vars = [1.48e-5, 1.76e-5, 1.62e-5]

plt.figure(figsize=(9, 6))
bars = plt.bar(models, f1_scores, yerr=f1_stds, capsize=10, color=['#2ca02c', '#d62728', '#1f77b4'], alpha=0.85, edgecolor='black', width=0.45)
plt.title("Supervised Models 5-Fold CV Mean F1-Score with Std Dev Error Bars", fontsize=13, fontweight='bold')
plt.ylabel("Validation F1-Score")
plt.ylim(0.70, 0.76)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f"Mean: {f1_scores[i]:.4f}\n(s² = {f1_vars[i]:.2e})", ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('results/eda_visualizations/04_cv_f1_variance_errorbars.png', dpi=300)
plt.close()

# Copy summary files to results/outputs
ratings_df.head(100).to_csv('results/outputs/sample_processed_ratings.csv', index=False)
movies_feat.to_csv('results/outputs/processed_movie_features.csv', index=False)

print("[SUCCESS] All presentation figures generated in results/eda_visualizations/!")
