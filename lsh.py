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

    return final_dataset

def group():

    final_dataset = get_dataset()

    grouped = final_dataset.groupby('rotten_tomatoes_link')
    
    movie_data = []

    for movie_id, movie_reviews in grouped:

        reviews = movie_reviews['review_content'].tolist()

        # Get metadata (take first row since they're all the same)
        first_row = movie_reviews.iloc[0]

        #append dict of each movie
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

    vectorized_movie_data =  movie_data_grouped

    min_year = 1914
    max_year = 2020
    max_runtime = 266


    model = SentenceTransformer('all-MiniLM-L6-v2')

    #dictionary of each movie 
    for movie in vectorized_movie_data:

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

    print(vectorized_movie_data[0])
    return vectorized_movie_data

#SimHash converts high-dimensional vectors into compact binary fingerprints (e.g., 64-bit integers).
def vector_to_simhash(vectorized_movie_data:list,bits=64):

    #gets the dimension of final vector
    dimension = len(vectorized_movie_data[0]['final_vector'])

    #creates 64 hyperplanes of dimension size
    hyperplanes = [np.random.randn(dimension) for _ in range(bits)]

    #default dict so we can store keys for hashes. identify hashes later.
    simhash_signatures = defaultdict(list)

    #Creates simhash of size bits
    for movie in vectorized_movie_data:

        hash_val = 0
        vector = movie['final_vector']

        for i,hp in enumerate(hyperplanes):
            if np.dot(vector, hp) > 0:
                hash_val |= (1 << i)#not sure this how works
        simhash_signatures[movie['movie_id']].append(hash_val)

    # print(simhash_signatures)
    return simhash_signatures

def build_lsh(simhash_signatures: defaultdict, bands = 8, bits =64):

    #rows = bits per band
    rows = bits // bands
    #mask is used to extract bands
    mask = (1 << rows) -1

    
    tables = [defaultdict(list) for _ in range(bands)]

    for movie_id in simhash_signatures:

        #hash corresponding to movie id
        hash = simhash_signatures[movie_id][0]

        # Distribute each movie into LSH hash tables based on its band-specific hash value
        for i,table in enumerate(tables):

            band_hash = (hash >> i*8) & mask
            table[band_hash].append(movie_id) 

    print(tables[0])
    return tables
    


#number of differing bits between two hashes. Smaller distance = more similar movies.
def hamming(h1, h2):
    return bin(h1 ^ h2).count('1')



def get_movie_name(movie_data, movie_id):
    
    for movie in movie_data:
        if movie['movie_id'] == movie_id:
            print(movie['movie_title'])

def get_movie_id(movie_data, movie_name):
    
    for movie in movie_data:
        if movie['movie_title'] == movie_name:
            print(movie['movie_id'])
    

# def build_lsh(simhash_signatures: defaultdict, bands = 8, bits =64):

#     rows = bits // bands
    
#     tables = [defaultdict(list) for _ in range(bands)]

#     for movie_id, hashes in simhash_signatures.items():
#         h = hashes[0]  # each movie has one hash
#         for i in range(bands):
#             start = i * rows
#             # extract bits for this band
#             band_hash = (h >> start) & ((1 << rows) - 1)
#             tables[i][band_hash].append(movie_id)
    
#     return tables

# def query_lsh(query_hash, tables, bands=8, n_bits=64):
#     rows = n_bits // bands
#     candidates = set()
#     for i in range(bands):
#         start = i * rows
#         band_hash = (query_hash >> start) & ((1 << rows) - 1)
#         candidates.update(tables[i][band_hash])
#     return candidates

   


def main():

    print("Loading and processing movie data...")
    
    grouped = group()

    vectorized_movie_data = vectorize(grouped) #list of dictionaries

    print("Creating Hashes...")

    simhash_signatures = vector_to_simhash(vectorized_movie_data)

    hash_tables = build_lsh(simhash_signatures)

    print("Querying...")

    movie = 'm/10005104-intruder'
    movie_name = "the intruder (l'intrus)"
    get_movie_name(vectorized_movie_data, movie)
    get_movie_id(vectorized_movie_data, movie_name)


    # print(len(movie_data))

if __name__ == "__main__":
    main()