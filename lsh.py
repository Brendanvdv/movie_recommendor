import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def get_dataset():
    final_dataset = pd.read_csv('final_dataset.csv')
    return final_dataset

def group():

    final_dataset = get_dataset()

    grouped = final_dataset.groupby('rotten_tomatoes_link')

    # print(grouped['movie_title'].first())
    
    movie_data = []

    for movie_id, movie_reviews in grouped:

        reviews = movie_reviews['review_content'].tolist()

        # Get metadata (take first row since they're all the same)
        first_row = movie_reviews.iloc[0]

        movie_data.append({
            'movie_id': movie_id,
            'movie_title': first_row['movie_title'],
            'reviews': reviews,  # List of all review texts
            'runtime': first_row['runtime'],
            'year': int((first_row['original_release_date']).split('-')[0]),
            'tomatometer': first_row['tomatometer_rating'],
            'genres': first_row['genres']
        })

    
    # print(len(movie_data))
    # print(movie_data[85]['year'])

    #keep movies above minimum amount of reviews
    min_reviews = 5

    movie_data_filtered = [
        movie for movie in movie_data
        if len(movie['reviews']) >= min_reviews
    ]


    # for movie in movie_data_filtered:
    #     num = len(movie['reviews'])
    #     if len(movie['reviews']) < num:
    #         num = len(movie['reviews'])

    # print(num)

    return movie_data_filtered

def vectorize(movie_data_filtered: list):

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for movie in movie_data_filtered:
        # Encode all reviews for this movie
        review_embeddings = model.encode(movie['reviews'])# Shape: (num_reviews, 384)


        #Average to get single movie embedding
        movie['review_embeddings'] = review_embeddings.mean(axis=0)
        # print(movie['review_embeddings']) 
    


def main():
    group()
    vectorize(group())

if __name__ == "__main__":
    main()