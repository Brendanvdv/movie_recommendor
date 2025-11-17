import numpy as np
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import seaborn as sns



#Global Variable------------------------------------------------------------------------------------

all_genres = [
    'science fiction & fantasy', 'drama', 'western', 'comedy', 'classics',
    'action & adventure', 'kids & family', 'musical & performing arts',
    'documentary', 'art house & international', 'horror', 'sports & fitness',
    'faith & spirituality', 'mystery & suspense', 'animation', 'special interest', 'romance'
]
#no 'nr' to make it a neutral vector if its in the rating
all_age_ratings = ['pg', 'r', 'g', 'pg-13', 'nc17']


#Movie Data-------------------------------------------------------------------------------------------

def get_dataset() -> pd.DataFrame:

    dataset = 'final_dataset.csv'

    final_dataset = pd.read_csv(dataset)

    return final_dataset

def group() -> list[dict]:

    final_dataset = get_dataset()
    
    #group movies together
    grouped = final_dataset.groupby('rotten_tomatoes_link')

    movie_data = []

    for movie_id, group in grouped:

        #Critic data
        reviews = group['review_content'].tolist()
        review_scores = group['review_score'].tolist()
        critic_names = group['critic_name'].to_list()
        review_types = group['review_type'].to_list()

        # Get metadata (take first row since they're all the same)
        first_row = group.iloc[0]

        #append dict of each movie
        movie_data.append({
            'movie_id': movie_id,
            'movie_title': first_row['movie_title'],
            'content_rating': first_row['content_rating'],
            'genres': first_row['genres'],
            'year': first_row['original_release_date'],
            'movie_info': first_row['movie_info'], #List of all review info texts

            #Per critic data
            'reviews': reviews,  # List of all review texts
            'review_scores': review_scores, #List of all review scores
            'critic_names': critic_names,   #List of all critic names
            'review_types': review_types, #List of all review types
            
            #Info
            # 'avg_review_type': np.mean(review_types), 
            'avg_score': np.mean(review_scores),
            'num_reviews': len(reviews)

        })
        
    #minimum amount of reviews
    min_reviews = 5

    movie_data_filtered = [
        movie for movie in movie_data
        if len(movie['reviews']) >= min_reviews
    ]
    
    print(f"Loaded {len(movie_data_filtered)} movies with ≥{min_reviews} reviews")
    movie_data = movie_data_filtered

    return movie_data


#Vectorize Data-----------------------------------------------------------------------------------

#Create embeddings for content and review style
def vectorize(movie_data: list) -> list[dict]:

    #creates binary list of genres 
    def encode_genres(movie_genres:list,all_genres:list):
        return [1 if genre in movie_genres else 0 for genre in all_genres]
    
    #creates binary list of ratings
    def encode_content_rating(content_rating,all_content_ratings):
        return [1 if cr == content_rating else 0 for cr in all_content_ratings]
    
    def encode_review_type(review_types):
        return [1 if rt == "fresh" else 0 for rt in review_types]
       
    print("Initializing sentence transformer...")
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')#faster but worse perfomance
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    #normalize year range
    year = [movie['year'] for movie in movie_data if movie['year'] > 0]
    min_year = min(year)
    max_year = max(year)

    total_movies = len(movie_data)

    for idx, movie in enumerate(movie_data, start=1):

        # Print progress
        print(f"Processing movie {idx}/{total_movies} ({idx/total_movies*100:.1f}%) - {movie['movie_title']}")

        ###################
        #encode movie info:
        ###################

        # Encode all movie infos for this movie
        movie['movie_info_embeddings'] = model.encode(movie['movie_info'])

        ########################
        #vectorize/norm metadata:
        ########################

        #Content rating
        movie['content_rating_norm'] = encode_content_rating(movie['content_rating'],all_age_ratings)

        #Genres
        movie['genre_vector'] = encode_genres(movie['genres'], all_genres)

        #Year
        movie['year_norm'] = (movie['year'] - min_year) / (max_year - min_year)

        ##############
        #content vector:
        ##############

        movie['content_vector'] = np.concatenate([
            np.array(movie['movie_info_embeddings']),
            np.array(movie['content_rating_norm']),
            np.array(movie['genre_vector']),
            np.array([ movie['year_norm']])
        ])
        
        ################
        #encode reviews:
        ################

        # Encode all reviews for this movie
        review_embeddings = model.encode(movie['reviews'])# Shape: (num_reviews, 384)

        ################
        #review style embedded vector:
        ################

        #Average to get single movie embedding
        movie['avg_review_embeddings'] = review_embeddings.mean(axis=0)

        ################
        #review score
        ################

        #Review rating
        movie['avg_score_norm']= movie['avg_score']/100

        ########
        #Review Type
        ########

        movie['review_types_norm'] = encode_review_type(movie['review_types'])
       
    vectorized_movie_data = movie_data

    print(vectorized_movie_data[0])
    return vectorized_movie_data

#checks Load vectorized movie data from pickle if it exists. Otherwise, generate, save, and return it
def load_or_create_vectorized_data(pickle_path="vectorized_movie_data.pkl") -> list[dict]:
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


#Similarity-----------------------------------------------------------------------------------------

#content
def build_content_sim(vectorized_movie_data) -> np.ndarray:

    content_vectors = np.vstack([movie['content_vector'] for movie in vectorized_movie_data])

    #pairwise cosine similarity
    content_sim = cosine_similarity(content_vectors).astype('float32')

    print(f"Content similarity matrix: {content_sim.shape}")

    # print(content_sim)
    return content_sim
    
#style
def build_review_style_sim(vectorized_movie_data) -> np.ndarray:
    
    review_embeddings = np.vstack([movie['avg_review_embeddings'] for movie in vectorized_movie_data])

    #pairwise cosine similarity
    review_sim = cosine_similarity(review_embeddings).astype('float32')

    print(f"Review similarity matrix: {review_sim.shape}")

    # print(review_sim)
    return review_sim

#taste
def build_review_score_sim(vectorized_movie_data):

    #Get all critics
    all_critics = set()
    for movie in vectorized_movie_data:
        all_critics.update(movie['critic_names'])
    all_critics = sorted(all_critics)

    #Get all movie titles
    all_movies = [movie['movie_title'] for movie in vectorized_movie_data]
    

    #create Movie x Critic matrix
    matrix = pd.DataFrame(np.nan, index = all_movies, columns=all_critics)

    #Fill in review scores
    for movie in vectorized_movie_data:

        mt = movie['movie_title']

        for critic,score in zip(movie['critic_names'], movie['review_scores']):
            matrix.loc[mt,critic] = score

    #Pearson correlation movie x movie (1 to -1)
    score_sim = matrix.T.corr(method='pearson')

    # normalize to 0-1
    score_sim = (score_sim + 1) / 2

    #convert to np array
    score_sim_array = score_sim.values

    #fill empty pairs with 0.5(neutral)
    score_sim = np.nan_to_num(score_sim_array, nan=0.5)

    print(f"Content similarity matrix: {score_sim.shape}")

    return score_sim


def build_review_type_sim(vectorized_movie_data) -> np.ndarray:

    #Get all critics
    all_critics = set()
    for movie in vectorized_movie_data:
        all_critics.update(movie['critic_names'])
    all_critics = sorted(all_critics)

    #Get all movie titles
    all_movies = [movie['movie_title'] for movie in vectorized_movie_data]
    
    #create Movie x Critic matrix
    matrix = pd.DataFrame(np.nan, index = all_movies, columns=all_critics)

    #Fill in review scores
    for movie in vectorized_movie_data:

        mit = movie['movie_title']

        for critic,review_type in zip(movie['critic_names'], movie['review_types_norm']):
            matrix.loc[mit,critic] = review_type

    #Pearson correlation movie x movie (1 to -1)
    type_sim = matrix.T.corr(method='pearson')

    # normalize to 0-1
    type_sim = (type_sim + 1) / 2

    #convert to np array
    type_sim_array = type_sim.values

    #fill empty pairs with 0.5(neutral)
    type_sim = np.nan_to_num(type_sim_array, nan=0.5)

    print(f"Type similarity matrix: {type_sim.shape}")

    print(type_sim)
    return type_sim

def load_or_create_similarity_matrices(vectorized_data, pickle_path="similarity_matrices.pkl"):
    """Load or compute all similarity matrices"""
    
    if os.path.exists(pickle_path):
        print(f"Loading similarity matrices from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            matrices = pickle.load(f)
        content_sim = matrices['content']
        style_sim = matrices['style']
        type_sim = matrices['type']
    else:
        print("Pickle file not found. Computing similarity matrices...")
        content_sim = build_content_sim(vectorized_data)
        style_sim = build_review_style_sim(vectorized_data)
        type_sim = build_review_type_sim(vectorized_data)
        
        print(f"Saving similarity matrices to {pickle_path}...")
        with open(pickle_path, "wb") as f:
            pickle.dump({
                'content': content_sim,
                'style': style_sim,
                'type': type_sim
            }, f)
    
    return content_sim, style_sim, type_sim
#Query--------------------------------------------------------------------------------------------------

def hybrid_score(content_sim,review_sim,type_sim,alpha, beta, gamma):
    
    hybrid_sim = (alpha * content_sim) + (beta * review_sim) + (gamma * type_sim)

    return hybrid_sim

def query_movie(movie_title,vectorized_movie_data,hybrid_sim,k=10):

    #Find iterable of query movie
    query_id = None
    for i,movie in enumerate(vectorized_movie_data):
        if movie['movie_title'] == movie_title:
            query_id = i
            break
    
    #row of hybrid sim scores sorted
    sim = hybrid_sim[query_id]
    sorted_indices = np.argsort(sim)[::-1]

    recommendations = []

    for pos in sorted_indices[1:k+1]:
        recommendations.append((vectorized_movie_data[pos]['movie_title'], sim[pos]))

    ################################
    max_len = max(len(movie) for movie, _ in recommendations)
    print()
    print(f"Query movie: {movie_title}")
    print()
    print("MOVIE".ljust(max_len)  +  "  SCORE")

    for m,s in recommendations:
        score = round(s,2)
        print(f'{m:{max_len}}  {score}')
       
    return recommendations

#----------------------------------------------------------------------------------------------------------

def main():

    vectorized_data = load_or_create_vectorized_data()

    content_sim, style_sim, type_sim = load_or_create_similarity_matrices(vectorized_data)
    hybrid_sim = hybrid_score(content_sim, style_sim, type_sim, alpha=0.5, beta=0.4, gamma=0.1)
    
    query_title = "the shining"

    query_movie(query_title, vectorized_data, hybrid_sim, k=10)

if __name__ == "__main__":
    main()