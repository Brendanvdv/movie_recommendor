import pandas as pd
import numpy as np

df_critic_reviews = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
df_movies = pd.read_csv('rotten_tomatoes_movies.csv')

#combines the two datasets
def merge_datasets():

    df_merged = pd.merge(df_critic_reviews,df_movies,on='rotten_tomatoes_link', how='inner')
    return df_merged
    
    
    
#chooses columns we want and filters out anything we dont want
def fix_filter_dataset(df: pd.DataFrame):

    #remove empty rows
    df_fixed = df.dropna(subset=['review_content','original_release_date','genres','runtime'])
    

    unwanted_columns = [
        'critic_name',
        'top_critic',
        'publisher_name',
        'movie_info',
        'critics_consensus',
        'content_rating',
    ]
    
    #get rid of unwanted columns
    df_filtered = df_fixed.drop(unwanted_columns, axis='columns')

    #only fresh(positive) reviews
    df_filtered = df_filtered[df_filtered['review_type'] == 'fresh']

    #lowercase entire dataframe
    df_filtered = df_filtered.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

    

    return df_filtered

def create_csv(df: pd.DataFrame):

    df[0:1000].to_csv('final_dataset.csv')
    # df.to_csv('final_dataset.csv')



def main():

    # df_merged = merge_datasets()
    # df_ff = fix_filter_dataset(df_merged)
    # create_csv(df_ff)

    create_csv(fix_filter_dataset(merge_datasets()))

if __name__ == "__main__":
    main()