import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import defaultdict

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

    dataset = 'final_dataset.csv'

    final_dataset = pd.read_csv(dataset)

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

def group():

    final_dataset = get_dataset()

    grouped = final_dataset.groupby('rotten_tomatoes_link')
    
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
            'year': int((first_row['original_release_date']).split('-')[0]),#takes only year from date
            'critic_rating': first_row['tomatometer_rating'],
            'audience_rating': first_row['audience_rating'],
            'genres': first_row['genres']
        })

    
    # print(len(movie_data))
    # print(movie_data[85]['year'])

    #keep movies above minimum amount of reviews
    min_reviews = 5

    movie_data_grouped = [
        movie for movie in movie_data
        if len(movie['reviews']) >= min_reviews
    ]

    #check min movie reviews
    # for movie in movie_data_grouped:
    #     num = len(movie['reviews'])
    #     if len(movie['reviews']) < num:
    #         num = len(movie['reviews'])

    # print(num)
   

    return movie_data_grouped

#creates binary vector of genres 
def encode_genres(movie_genres:list,all_genres:list):
    return [1 if genre in movie_genres else 0 for genre in all_genres]

def vectorize(movie_data_grouped: list):

    min_year = 1914
    max_year = 2020
    max_runtime = 266


    model = SentenceTransformer('all-MiniLM-L6-v2')

    #dictionary of each movie 
    for movie in movie_data_grouped:

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

        #Critic rating
        movie['critic_rating_norm'] = movie['critic_rating']/100

        #Audience rating
        movie['audience_rating_norm'] = movie['audience_rating']/100

        #Genres
        all_genres = all_g
        movie['genre_vector']= encode_genres(movie['genres'], all_genres)

        #year
        movie['year_norm'] = (movie['year'] - min_year) / (max_year - min_year)

        #runtime
        movie['runtime_norm'] = movie['runtime']/max_runtime

        ##############
        #Final vector:
        ##############
        
        final_vector = np.concatenate([
            np.array(movie['review_embeddings']),
            np.array([movie['critic_rating_norm']]),
            np.array([movie['audience_rating_norm']]),
            np.array([ movie['year_norm']]),
            np.array(movie['genre_vector']),
            np.array([movie['runtime_norm']])
        ])
        
        movie['final_vector'] = final_vector

    print(movie_data_grouped[0])
    return movie_data_grouped

#SimHash converts high-dimensional vectors into compact binary fingerprints (e.g., 64-bit integers).
def vector_to_simhash(movie_data_grouped:list,bits=64):

    #gets the dimension of final vector
    dimension = len(movie_data_grouped[0]['final_vector'])

    #creates 64 hyperplanes of dimension size
    hyperplanes = [np.random.randn(dimension) for _ in range(bits)]

    hash_val = 0
    simhashes = []

    #Creates simhash of size bits
    for movie in movie_data_grouped:

        vector = movie['final_vector']

        for i,hp in enumerate(hyperplanes):
            if np.dot(vector, hp) > 0:
                hash_val |= (1 << i)#not sure this how works
        simhashes.append(hash_val)


    return simhashes

#number of differing bits
def hamming(h1, h2):
    return bin(h1 ^ h2).count('1')

def build_lsh(movie_data_grouped:list):
    pass


    # movie_ids = [movie['movie_id'] for movie in movie_data_grouped]
    # movie_vectors = np.array([movie['final_vector'] for movie in movie_data_grouped])
    # print(len(movie_vectors[0]))


def main():

    print("Loading and processing movie data...")
    
    grouped = group()

    movie_data = vectorize(grouped)

    print("Creating Hashes...")

    vector_to_simhash(movie_data)

    # print(len(movie_data))

if __name__ == "__main__":
    main()