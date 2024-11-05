# assignment_1-AIE425
Intelligent Recommender Systems assignment_1

Overview
This project is about building a recommender system using User-Based and Item-Based Collaborative Filtering. i scraped IMDb's Top 250 movies to get the data and taked only 10 column and 50 raw, and used cosine similarity and Pearson correlation to predict ratings for unrated items.

Features
Collaborative Filtering: User-based and item-based methods to recommend items.
Similarity Measures: Using cosine similarity and Pearson correlation.
Data Collection: Data scraped from IMDb.
Predictions: Generated based on similar users or items.
Comparison: Evaluated the effectiveness of both CF methods.

Technologies Used
Python: Main programming language.
Requests & BeautifulSoup: For web scraping IMDb.
Pandas & NumPy: For data handling and analysis.
Scikit-learn & SciPy: For similarity calculations.

How to Use
Clone: git clone <repo-url>.
Install Requirements : libraries 
Run Script: Execute code.py to run the system.
View Results: Predicted ratings will be displayed and can be saved as a CSV.

Results Summary
Cosine Similarity: Consistently high similarity scores.
Pearson Correlation: Effective but struggled with sparse data.
User-Based vs Item-Based: Item-based was more stable, user-based offered more personalization.

Future Enhancements
Hybrid Filtering: Combine user and item-based methods.
Contextual Data: Add user demographics and movie genres.
Deep Learning: Use neural collaborative filtering for better predictions.



