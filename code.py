import requests
import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from bs4 import BeautifulSoup

# URL for IMDB's Top 250 movies
url = "https://www.imdb.com/chart/top"

# Fetch movie titles and ratings from IMDB
response = requests.get(url, headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
})

if response.status_code != 200:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
    exit()

soup = BeautifulSoup(response.content, 'html.parser')

# Find and extract JSON data embedded in the script tags
scripts = soup.find_all('script', type='application/ld+json')
movie_data = []

for script in scripts:
    try:
        data = json.loads(script.string)
        if "itemListElement" in data:
            movie_data = data["itemListElement"]
            break
    except (json.JSONDecodeError, TypeError):
        continue

# Debug: Check if data was extracted
if not movie_data:
    print("No movie data found in the embedded JSON structure.")
    exit()

# Extract movie titles and ratings
movie_titles = []
movie_ratings = []
for item in movie_data[:10]:  # Limit to 10 movies
    movie = item.get('item', {})
    title = movie.get('name')
    rating = movie.get('aggregateRating', {}).get('ratingValue')
    if title:
        movie_titles.append(title)
        movie_ratings.append(int(rating) if rating else None)  # Keep missing ratings as None

# Ensure movie data is correctly captured
if not movie_titles:
    print("Movie titles not found. Please check the JSON extraction logic.")
    exit()

# Generate a list of users
users = [f'User_{i+1}' for i in range(50)]  # 50 users

# Save data to CSV
with open('imdb_user_ratings.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    header = ['User'] + movie_titles
    writer.writerow(header)

    for user in users:
        user_ratings = []
        for rating in movie_ratings:
            if rating is not None:
                if np.random.rand() > 0.1:  # 10% chance to introduce a missing value
                    # Add a small random variation to each user's rating
                    varied_rating = round(rating + np.random.uniform(-1, 1), 1)
                    # Ensure the rating stays within a valid range (1 to 10)
                    varied_rating = int(max(1, min(10, varied_rating)))
                    user_ratings.append(varied_rating)
                else:
                    user_ratings.append(None)  # Introduce missing value
            else:
                user_ratings.append(None)  # Maintain missing values
        writer.writerow([user] + user_ratings)

print("Data has been saved to imdb_user_ratings.csv")

# Read the CSV file into a DataFrame
df = pd.read_csv('imdb_user_ratings.csv')

# Convert the DataFrame to a NumPy matrix (excluding the 'User' column)
ratings_matrix = df.drop(columns=['User']).to_numpy()

# Replace NaNs with zeros for similarity calculations
ratings_matrix = np.nan_to_num(ratings_matrix)

# Compute cosine similarity for users to identify peer groups
user_cosine_similarity = cosine_similarity(ratings_matrix)
print("User-based Cosine Similarity:\n", user_cosine_similarity)

# Compute cosine similarity for items to identify peer groups
item_cosine_similarity = cosine_similarity(ratings_matrix.T)
print("Item-based Cosine Similarity:\n", item_cosine_similarity)

# Compute Pearson correlation for users to identify peer groups
def pearson_similarity(matrix):
    num_users = matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(num_users):
            if i != j:
                valid_indices = (matrix[i] != 0) & (matrix[j] != 0)
                if np.sum(valid_indices) > 1:
                    similarity_matrix[i, j], _ = pearsonr(matrix[i, valid_indices], matrix[j, valid_indices])
                else:
                    similarity_matrix[i, j] = 0
            else:
                similarity_matrix[i, j] = 1
    similarity_matrix[similarity_matrix < 0] = 0  # Set negative correlations to 0
    return similarity_matrix

user_pearson_similarity = pearson_similarity(ratings_matrix)
print("User-based Pearson Correlation:\n", user_pearson_similarity)

# Compute Pearson correlation for items to identify peer groups
def item_pearson_similarity(matrix):
    num_items = matrix.shape[1]
    similarity_matrix = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(num_items):
            if i != j:
                valid_indices = (matrix[:, i] != 0) & (matrix[:, j] != 0)
                if np.sum(valid_indices) > 1:
                    similarity_matrix[i, j], _ = pearsonr(matrix[valid_indices, i], matrix[valid_indices, j])
                else:
                    similarity_matrix[i, j] = 0
            else:
                similarity_matrix[i, j] = 1
    similarity_matrix[similarity_matrix < 0] = 0  # Set negative correlations to 0
    return similarity_matrix

item_pearson_similarity = item_pearson_similarity(ratings_matrix)
print("Item-based Pearson Correlation:\n", item_pearson_similarity)

# Rating prediction function
def predict_ratings(sim_matrix, ratings_matrix, type='user'):
    predictions = np.zeros(ratings_matrix.shape)
    if type == 'user':
        for i in range(ratings_matrix.shape[0]):
            for j in range(ratings_matrix.shape[1]):
                if ratings_matrix[i, j] == 0:  # Only predict for missing ratings
                    sim_scores = sim_matrix[i]
                    ratings = ratings_matrix[:, j]
                    valid_ratings = (ratings != 0) & (sim_scores != 0)
                    if np.sum(valid_ratings) > 0:
                        predictions[i, j] = np.dot(sim_scores[valid_ratings], ratings[valid_ratings]) / np.sum(np.abs(sim_scores[valid_ratings]))
    elif type == 'item':
        for i in range(ratings_matrix.shape[0]):
            for j in range(ratings_matrix.shape[1]):
                if ratings_matrix[i, j] == 0:  # Only predict for missing ratings
                    sim_scores = sim_matrix[j]
                    ratings = ratings_matrix[i, :]
                    valid_ratings = (ratings != 0) & (sim_scores != 0)
                    if np.sum(valid_ratings) > 0:
                        predictions[i, j] = np.dot(sim_scores[valid_ratings], ratings[valid_ratings]) / np.sum(np.abs(sim_scores[valid_ratings]))
    return predictions

# Predict ratings for user-based CF using cosine similarity
user_cosine_predictions = predict_ratings(user_cosine_similarity, ratings_matrix, type='user')
print("Predicted Ratings (User-based CF, Cosine Similarity):\n", user_cosine_predictions)

# Predict ratings for item-based CF using cosine similarity
item_cosine_predictions = predict_ratings(item_cosine_similarity, ratings_matrix, type='item')
print("Predicted Ratings (Item-based CF, Cosine Similarity):\n", item_cosine_predictions)

# Predict ratings for user-based CF using Pearson correlation
user_pearson_predictions = predict_ratings(user_pearson_similarity, ratings_matrix, type='user')
print("Predicted Ratings (User-based CF, Pearson Correlation):\n", user_pearson_predictions)

# Predict ratings for item-based CF using Pearson correlation
item_pearson_predictions = predict_ratings(item_pearson_similarity, ratings_matrix, type='item')
print("Predicted Ratings (Item-based CF, Pearson Correlation):\n", item_pearson_predictions)

with open('predicted_ratings_report.txt', 'w') as f:
    f.write("Predicted Ratings (User-based CF, Cosine Similarity):\n")
    np.savetxt(f, user_cosine_predictions, fmt='%.2f')
    f.write("\n\nPredicted Ratings (Item-based CF, Cosine Similarity):\n")
    np.savetxt(f, item_cosine_predictions, fmt='%.2f')
    f.write("\n\nPredicted Ratings (User-based CF, Pearson Correlation):\n")
    np.savetxt(f, user_pearson_predictions, fmt='%.2f')
    f.write("\n\nPredicted Ratings (Item-based CF, Pearson Correlation):\n")
    np.savetxt(f, item_pearson_predictions, fmt='%.2f')

# Get top-N recommendations for a given user
def get_top_n_recommendations(predictions, user_index, n=5):
    user_ratings = predictions[user_index]
    top_n_items = user_ratings.argsort()[-n:][::-1]
    return top_n_items

# Get top-5 recommendations for User_1 using different methods
user_index = 20
print("Top-5 Recommendations for User_1 (User-based CF, Cosine Similarity):", get_top_n_recommendations(user_cosine_predictions, user_index))
print("Top-5 Recommendations for User_1 (Item-based CF, Cosine Similarity):", get_top_n_recommendations(item_cosine_predictions, user_index))
print("Top-5 Recommendations for User_1 (User-based CF, Pearson Correlation):", get_top_n_recommendations(user_pearson_predictions, user_index))
print("Top-5 Recommendations for User_1 (Item-based CF, Pearson Correlation):", get_top_n_recommendations(item_pearson_predictions, user_index))

# Save matrices to a text file for report
with open('similarity_matrices_report.txt', 'w') as f:
    f.write("User-based Cosine Similarity:\n")
    np.savetxt(f, user_cosine_similarity, fmt='%.4f')
    f.write("\n\nItem-based Cosine Similarity:\n")
    np.savetxt(f, item_cosine_similarity, fmt='%.4f')
    f.write("\n\nUser-based Pearson Correlation:\n")
    np.savetxt(f, user_pearson_similarity, fmt='%.4f')
    f.write("\n\nItem-based Pearson Correlation:\n")
    np.savetxt(f, item_pearson_similarity, fmt='%.4f')

print("Similarity matrices have been saved to similarity_matrices_report.txt")
 # type: ignore