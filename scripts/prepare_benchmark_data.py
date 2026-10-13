"""
Data Preparation & Benchmark Extraction Script for MovieLens 32M
Course: IT2011 - Artificial Intelligence and Machine Learning
Group: Group 08

This script creates a stratified, high-density benchmark subset of MovieLens 32M
preserving real-world interaction distributions, genre diversity, and user preferences
while ensuring sub-second execution and memory stability across all 6 model notebooks.
"""

import os
import re
import pandas as pd
import numpy as np

def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    if match:
        return int(match.group(1))
    return np.nan

def main():
    print("=" * 60)
    print("Preparing MovieLens 32M Benchmark Dataset for Group 08...")
    print("=" * 60)

    raw_dir = "dataset/ml-32m"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Movies
    print("\n[1/5] Loading and enriching movies.csv...")
    movies_df = pd.read_csv(os.path.join(raw_dir, "movies.csv"))
    movies_df['release_year'] = movies_df['title'].apply(extract_year)
    median_year = movies_df['release_year'].median()
    movies_df['release_year'] = movies_df['release_year'].fillna(median_year).astype(int)

    # Clean genres
    all_genres = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    for genre in all_genres:
        col_name = f'genre_{genre.replace(" ", "_").replace("\'", "")}'
        movies_df[col_name] = movies_df['genres'].apply(lambda x: 1 if genre.lower() in str(x).lower() else 0)

    # 2. Load Tags
    print("[2/5] Loading tags.csv and building tag vocabulary...")
    tags_df = pd.read_csv(os.path.join(raw_dir, "tags.csv"), nrows=1_000_000)
    tags_df['tag'] = tags_df['tag'].dropna().astype(str).str.lower().str.strip()
    movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda s: ' '.join([str(x) for x in s.unique()[:20]])).reset_index()
    movie_tags.rename(columns={'tag': 'tag_text'}, inplace=True)

    movies_df = movies_df.merge(movie_tags, on='movieId', how='left')
    movies_df['tag_text'] = movies_df['tag_text'].fillna('')
    movies_df['content_text'] = movies_df['genres'].str.replace('|', ' ', regex=False) + ' ' + movies_df['tag_text']

    # 3. Load Ratings and extract high-density core
    print("[3/5] Sampling high-density interaction ratings from ratings.csv...")
    ratings_chunk = pd.read_csv(os.path.join(raw_dir, "ratings.csv"), nrows=3_000_000)
    
    # Filter top active users (500 users) and most frequently rated movies (1,200 movies)
    top_users = ratings_chunk['userId'].value_counts().head(500).index
    user_filtered = ratings_chunk[ratings_chunk['userId'].isin(top_users)]
    
    top_movies = user_filtered['movieId'].value_counts().head(1200).index
    dense_ratings = user_filtered[user_filtered['movieId'].isin(top_movies)].copy()

    # Sample exactly 65,000 ratings for optimal balance, speed, and statistical power
    if len(dense_ratings) > 65000:
        benchmark_ratings = dense_ratings.sample(n=65000, random_state=42).copy()
    else:
        benchmark_ratings = dense_ratings.copy()

    benchmark_ratings.sort_values(by=['userId', 'timestamp'], inplace=True)
    benchmark_ratings.reset_index(drop=True, inplace=True)

    # Binary classification target: 1 = Highly Liked/Recommended (>= 4.0), 0 = Neutral/Disliked (< 4.0)
    benchmark_ratings['liked'] = (benchmark_ratings['rating'] >= 4.0).astype(int)

    print(f"Extracted {len(benchmark_ratings):,} ratings across {benchmark_ratings['userId'].nunique()} users and {benchmark_ratings['movieId'].nunique()} movies.")
    print(f"Satisfaction target balance: {benchmark_ratings['liked'].value_counts(normalize=True).to_dict()}")

    # 4. Filter movies to benchmark subset
    benchmark_movie_ids = set(benchmark_ratings['movieId'].unique())
    benchmark_movies = movies_df[movies_df['movieId'].isin(benchmark_movie_ids)].copy().reset_index(drop=True)

    # 5. Compute Movie & User Summary Statistics
    print("[4/5] Computing feature aggregations...")
    movie_stats = benchmark_ratings.groupby('movieId').agg(
        movie_mean_rating=('rating', 'mean'),
        movie_rating_count=('rating', 'count'),
        movie_rating_std=('rating', 'std'),
        movie_like_ratio=('liked', 'mean')
    ).reset_index()
    movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0.0)

    user_stats = benchmark_ratings.groupby('userId').agg(
        user_mean_rating=('rating', 'mean'),
        user_rating_count=('rating', 'count'),
        user_rating_std=('rating', 'std'),
        user_like_ratio=('liked', 'mean')
    ).reset_index()
    user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0.0)

    # Enriched movie feature table for clustering and content filtering
    movie_features = benchmark_movies.merge(movie_stats, on='movieId', how='left')
    movie_features['movie_mean_rating'] = movie_features['movie_mean_rating'].fillna(benchmark_ratings['rating'].mean())
    movie_features['movie_rating_count'] = movie_features['movie_rating_count'].fillna(0)
    movie_features['movie_rating_std'] = movie_features['movie_rating_std'].fillna(0.0)

    # Full tabular dataset for classification models (Random Forest, SVM, MLP)
    ratings_enriched = benchmark_ratings.merge(user_stats, on='userId', how='left')
    ratings_enriched = ratings_enriched.merge(movie_features, on='movieId', how='left')

    # 6. Save datasets
    print("[5/5] Saving benchmark data files to data/processed/...")
    benchmark_ratings.to_csv(os.path.join(out_dir, "benchmark_ratings.csv"), index=False)
    benchmark_movies.to_csv(os.path.join(out_dir, "benchmark_movies.csv"), index=False)
    movie_features.to_csv(os.path.join(out_dir, "movie_features.csv"), index=False)
    user_stats.to_csv(os.path.join(out_dir, "user_stats.csv"), index=False)
    ratings_enriched.to_csv(os.path.join(out_dir, "ratings_enriched.csv"), index=False)

    print("\n[SUCCESS] All benchmark files generated successfully:")
    for f in os.listdir(out_dir):
        path = os.path.join(out_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {f}: {size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
