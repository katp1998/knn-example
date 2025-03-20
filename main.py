import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial import distance

# Load data:
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Process data (bring them to a common format):
movies['genres'] = movies['genres'].apply(json.loads).apply(lambda x: [i['name'] for i in x])
movies['keywords'] = movies['keywords'].apply(json.loads).apply(lambda x: [i['name'] for i in x])
movies['production_companies'] = movies['production_companies'].apply(json.loads).apply(lambda x: [i['name'] for i in x])
credits['cast'] = credits['cast'].apply(json.loads).apply(lambda x: [i['name'] for i in x])
credits['crew'] = credits['crew'].apply(json.loads).apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'])
credits.rename(columns={'crew': 'director'}, inplace=True)

# Merge datasets:
movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
movies = movies[['id', 'original_title', 'genres', 'cast', 'vote_average', 'director', 'keywords', 'budget']]
# print(movies['vote_average'])

# One-hot encode categorical data:
mlb = MultiLabelBinarizer()
movies['genres_bin'] = list(mlb.fit_transform(movies['genres']))

# Convert list columns to numpy arrays (to make it efficient):
movies['genres_bin'] = movies['genres_bin'].apply(lambda x: np.array(x))

# calculate euclidean distance:
def euclidean_similarity(movie1, movie2):
    genres_dist = distance.euclidean(movie1['genres_bin'], movie2['genres_bin'])
    return genres_dist

# Movie recommendation using SciPy:
def recommend_movies_scipy(movie_name, K=10):
    base_movie = movies[movies['original_title'].str.contains(movie_name, case=False)].iloc[0]
    distances = []
    for _, movie in movies.iterrows():
        if movie['id'] != base_movie['id']:
            dist = euclidean_similarity(base_movie, movie)
            distances.append((movie['original_title'], movie['genres'], movie['vote_average'], dist))
    
    distances.sort(key=lambda x: x[3])
    
    print(f"\nRecommendations using SciPy for: {base_movie['original_title']}\n")
    for rec in distances[:K]:
        print(f"{rec[0]} | Genres: {', '.join(rec[1])} | Rating: {rec[2]}")

# Movie recommendation using TensorFlow:
def recommend_movies_tensorflow(movie_name, K=10):
    base_movie = movies[movies['original_title'].str.contains(movie_name, case=False)].iloc[0]
    base_movie_features = np.array([base_movie['genres_bin']])

    X = np.array(list(movies['genres_bin']))

    #taining -- 3 layer neural networks:
    knn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(X.shape[1], activation='linear')
    ])

    knn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = knn_model.fit(X, X, epochs=10, batch_size=16, validation_split=0.2)

    plot_tensorflow_loss(history)
    
    distances = []
    predictions = knn_model.predict(X)
    for i, pred in enumerate(predictions):
        dist = np.linalg.norm(base_movie_features - pred)
        distances.append((movies.iloc[i]['original_title'], movies.iloc[i]['genres'], movies.iloc[i]['vote_average'], dist))
    
    distances.sort(key=lambda x: x[3])
    
    print(f"\nRecommendations using TensorFlow KNN for: {base_movie['original_title']}\n")
    for rec in distances[:K]:
        print(f"{rec[0]} | Genres: {', '.join(rec[1])} | Rating: {rec[2]}")

# EDA-- top genres:
def plot_top_genres():
    genre_list = []
    for genres in movies['genres']:
        genre_list.extend(genres)
    genre_counts = pd.Series(genre_list).value_counts()[:10]
    plt.figure(figsize=(10,6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.xlabel("Count")
    plt.ylabel("Genres")
    plt.title("Top 10 Genres")
    plt.show()

# EDA-- budget vs. ratings:
def plot_budget_vs_rating():
    plt.figure(figsize=(12,6))
    sns.scatterplot(x=movies['budget'], y=movies['vote_average'], alpha=0.6)

    plt.xlabel("Budget (in full value)")
    plt.ylabel("Vote Average")
    plt.title("Movie Budget vs. Ratings")
    plt.xscale('log')
    plt.show()

# EDA-- No of genres vs. ratings:
def plot_genres_vs_rating():
    plt.figure(figsize=(12,6))
    
    genre_rating_data = []
    for _, row in movies.iterrows():
        for genre in row['genres']:
            genre_rating_data.append((genre, row['vote_average']))

    genre_rating_df = pd.DataFrame(genre_rating_data, columns=['Genre', 'Vote Average'])
    sns.stripplot(x=genre_rating_df['Genre'], y=genre_rating_df['Vote Average'], jitter=True, alpha=0.6)
    
    plt.xlabel("Genre")
    plt.ylabel("Vote Average")
    plt.title("Genres vs. Movie Ratings")
    plt.xticks(rotation=45)
    plt.show()

# EDA-- TensorFlow Model Loss:
def plot_tensorflow_loss(history):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("TensorFlow Model Training Loss")
    plt.legend()
    plt.show()

# EDA-- Outliers:
def plot_outliers():
    plt.figure(figsize=(10,6))
    sns.boxplot(y=movies['vote_average'])
    plt.xlabel("Movies")
    plt.ylabel("Vote Average")
    plt.title("Interquartile Range (IQR) for Vote Averages to Detect Outliers")
    plt.show()
    

# Call all functions:
plot_top_genres()
plot_outliers()
plot_budget_vs_rating()
plot_genres_vs_rating()
recommend_movies_scipy("Inception")
recommend_movies_tensorflow("Inception")
