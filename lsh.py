import numpy as np
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict

all_g = [
    'science fiction & fantasy', 'drama', 'western', 'comedy', 'classics',
    'action & adventure', 'kids & family', 'musical & performing arts',
    'documentary', 'art house & international', 'horror', 'sports & fitness',
    'faith & spirituality', 'mystery & suspense', 'animation', 'special interest', 'romance'
]

all_r = ['pg', 'r', 'nr', 'g', 'pg-13', 'nc17']


def get_dataset():

    dataset = 'final_dataset.csv'

    final_dataset = pd.read_csv(dataset)

    return final_dataset

def group() -> list[dict]:

    final_dataset = get_dataset()
    

    #group movies together
    grouped = final_dataset.groupby('rotten_tomatoes_link')
    
    movie_data = []

    for movie_id, movie_columns in grouped:

        reviews = movie_columns['review_content'].tolist()

        # Get metadata (take first row since they're all the same)
        first_row = movie_columns.iloc[0]

        #average review score
        mean_review_score = movie_columns['review_score'].mean()

        #append dict of each movie
        movie_data.append({
            'movie_id': movie_id,
            'movie_title': first_row['movie_title'],
            'content_rating': first_row['content_rating'],
            'genres': first_row['genres'],
            'year': first_row['original_release_date'],
            'review_score': mean_review_score,
            'reviews': reviews,  # List of all review texts
            'movie_info': first_row['movie_info'] #List of all review info texts
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

#creates binary list of genres 
def encode_genres(movie_genres:list,all_genres:list):
    return [1 if genre in movie_genres else 0 for genre in all_genres]

#creates binary list of ratings
def encode_content_rating(content_ratings,all_content_ratings):
     return [1 if cr in content_ratings else 0 for cr in all_content_ratings]

def vectorize(movie_data_grouped: list) -> list[dict]:

    year = [movie['year'] for movie in movie_data_grouped if movie['year'] > 0]

    min_year = min(year)
    max_year = max(year)


    
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')#faster but worse perfomance
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    total_movies = len(movie_data_grouped)

    
    for idx, movie in enumerate(movie_data_grouped, start=1):

       
        # Print progress
        print(f"Processing movie {idx}/{total_movies} ({idx/total_movies*100:.1f}%) - {movie['movie_title']}")

        ################
        #encode reviews:
        ################

        # Encode all reviews for this movie
        review_embeddings = model.encode(movie['reviews'])# Shape: (num_reviews, 384)

        #Average to get single movie embedding
        movie['review_embeddings'] = review_embeddings.mean(axis=0)

        ###################
        #encode movie info:
        ###################

        # Encode all movie infos for this movie
        movie['movie_info_embeddings'] = model.encode(movie['movie_info'])

        ########################
        #vectorize/norm metadata:
        ########################

        #Content rating
        all_content_ratings = all_r
        movie['content_rating_norm'] = encode_content_rating(movie['content_rating'],all_content_ratings)

        #Review rating
        movie['review_score_norm'] = movie['review_score']/100

        #Genres
        all_genres = all_g
        movie['genre_vector'] = encode_genres(movie['genres'], all_genres)

        #Year
        movie['year_norm'] = (movie['year'] - min_year) / (max_year - min_year)

        ##############
        #Final vector:
        ##############

        final_vector = np.concatenate([
            np.array(movie['review_embeddings']),
            np.array(movie['movie_info_embeddings']),
            np.array(movie['content_rating_norm']),
            np.array([movie['review_score_norm']]),
            np.array(movie['genre_vector']),
            np.array([ movie['year_norm']])
        ])
        
        movie['final_vector'] = final_vector


    vectorized_movie_data = movie_data_grouped

    # print(vectorized_movie_data[0])
    return vectorized_movie_data

#checks Load vectorized movie data from pickle if it exists. Otherwise, generate, save, and return it
def load_or_create_vectorized_data(pickle_path="vectorized_movie_data.pkl"):
    if os.path.exists(pickle_path):
        print(f"Loading vectorized movie data from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            vectorized_movie_data = pickle.load(f)
    else:
        print("Pickle file not found. Generating vectorized movie data...")
        grouped = group()
        vectorized_movie_data = vectorize(grouped)
        with open(pickle_path, "wb") as f:
            pickle.dump(vectorized_movie_data, f)
        print(f"Vectorized movie data saved to {pickle_path}.")
    return vectorized_movie_data

#SimHash converts high-dimensional vectors into compact binary fingerprints (e.g., 64-bit integers).
def vector_to_simhash(vectorized_movie_data:list,bits=64) -> defaultdict[list]:

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

def build_lsh(simhash_signatures: defaultdict, bands = 8, bits =64): #list[defaultdict[int, list[str]]]

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

            band_hash = (hash >> i*bands) & mask
            table[band_hash].append(movie_id) 

    # print(tables[0])
    return tables

def get_movie_name(movie_data, movie_id) -> str:
    
    for movie in movie_data:
        if movie['movie_id'] == movie_id:
            # print(movie['movie_title'])
            return movie['movie_title']
    raise ValueError("Movie not found")

def get_movie_id(movie_data, movie_name) -> str:    
    
    for movie in movie_data:
        if movie['movie_title'] == movie_name:
            # print(movie['movie_id'])
            return movie['movie_id']
    raise ValueError("Movie not found")
   

#number of differing bits between two hashes. Smaller distance = more similar movies.
def hamming(h1, h2) -> int:
    return bin(h1 ^ h2).count('1')

def query_movie(query_movie_name,vectorized_movie_data,simhash_signatures,tables,n=10,bands = 8,bits = 64) -> list[tuple]:
    
        query_movie_id = get_movie_id(vectorized_movie_data, query_movie_name)
        query_hash = simhash_signatures[query_movie_id][0]

        # Set up parameters
        rows = bits // bands
        mask = (1 << rows) - 1

        # Collect candidates from all bands
        candidates = set()

        for band_num in range(bands):
            
            # Extract query's band signature
            shift_amount = band_num * rows
            band_signature = (query_hash >> shift_amount) & mask

            # Look up candidates in this band
            movies_in_bucket = tables[band_num][band_signature]
            candidates.update(movies_in_bucket)
            
        #remove query movie itself
        candidates.discard(query_movie_id)

        similarity_distance = []

        # Calculate hamming distance for each candidate
        for candidate_id in candidates:
            
            candidate_hash = simhash_signatures[candidate_id][0]

            distance = hamming(query_hash,candidate_hash)

            similarity_distance.append((candidate_id,distance))

        
        # Sort by distance and return top n
        similarity_distance.sort(key=lambda x: x[1])
        top_candidates = similarity_distance[:n]

        return top_candidates


def output_candidates(top_candidates,vectorized_movie_data):

    print("Similiar movies: ")
    print()

    for movie_id, distance in top_candidates:

        movie_name = get_movie_name(vectorized_movie_data,movie_id)

        print(f"{movie_name}: Distance = {distance}")
        print()


     




def main():

    

   # Load vectorized data from pickle or generate if missing
    vectorized_movie_data = load_or_create_vectorized_data()

    print("Creating Hashes...")

    simhash_signatures = vector_to_simhash(vectorized_movie_data)

    hash_tables = build_lsh(simhash_signatures)

    print("Querying...")

    # query_movie_name = "star wars: episode iii - revenge of the sith"
    query_movie_name = "10"

    top_candidates = query_movie(query_movie_name,vectorized_movie_data,simhash_signatures,hash_tables)

    output_candidates(top_candidates,vectorized_movie_data)

    # movie = 'm/star_wars_episode_iii_revenge_of_the_sith'
    # get_movie_name(vectorized_movie_data, movie)

    # movie_name = "the intruder (l'intrus)"
    # get_movie_id(vectorized_movie_data, movie_name)

    # matches = [movie['movie_title'] for movie in vectorized_movie_data if movie['movie_title'].lower() == query_movie_name.lower()]
    # print("Exact matches for query:", matches)



if __name__ == "__main__":
    main()

#ToDO
#Query



#Improve perfomance:
#Parameters
#check vector value range(normalize?)
#Increase bits in hyperplanes: 64 -> 128
#Use best sentence embed model
#change # of bands?

#Improve computation time:
#pickle file


#not sure:
#take average of review?
#nr(not rated) remove?
