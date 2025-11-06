import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

all_g = [
            'science fiction & fantasy',
            'drama',
            'western',
            'comedy',
            'classics',
            'action & adventure',
            'kids & family',
            'musical & performing arts',
            'documentary',
            'art house & international',
            'horror',
            'sports & fitness',
            'faith & spirituality',
            'mystery & suspense',
            'animation',
            'special interest',
            'romance'
            ]

def get_dataset():
    final_dataset = pd.read_csv('final_dataset.csv')

    # Get all unique genres from your dataset
    # all_genre_strings = final_dataset['genres'].dropna().unique()
    # # Example: ["Action, Sci-Fi", "Drama", "Comedy, Romance", ...]

    # # Parse and collect unique genres
    # unique_genres = set()
    # for genre_string in all_genre_strings:
    #     genres = genre_string.split(', ')
    #     unique_genres.update(genres)
    # print(unique_genres)

    return final_dataset

def encode_genres(movie_genres:list,all_genres:list):
    return [1 if genre in movie_genres else 0 for genre in all_genres]


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

    #check min movie reviews
    # for movie in movie_data_filtered:
    #     num = len(movie['reviews'])
    #     if len(movie['reviews']) < num:
    #         num = len(movie['reviews'])

    # print(num)

    return movie_data_filtered

def vectorize(movie_data_filtered: list):


    model = SentenceTransformer('all-MiniLM-L6-v2')

    #dictionary of each movie 
    for movie in movie_data_filtered:

        ################
        #encode reviews:
        ################

        # Encode all reviews for this movie
        review_embeddings = model.encode(movie['reviews'])# Shape: (num_reviews, 384)

        #Average to get single movie embedding
        movie['review_embeddings'] = review_embeddings.mean(axis=0)


        ########################
        #vectorize/norm metadata:
        ########################

        #Tomatometer
        movie['tomatometer_norm'] = movie['tomatometer']/100

        #Genres
        all_genres = all_g
        movie['genre_vector']= encode_genres(movie['genres'], all_genres)

        ##############
        #Final vector:
        ##############
        
        final_vector = np.concatenate([
            np.array(movie['review_embeddings']),
            np.array([movie['tomatometer_norm']]),
            np.array(movie['genre_vector'])
        ])

    print(final_vector)
    print(movie['tomatometer_norm'])


    # print(movie_data_filtered[0])



    

def main():
    
    grouped = group()

    vector = vectorize(grouped)

if __name__ == "__main__":
    main()