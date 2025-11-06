import pandas as pd
import numpy as np

df_critic_reviews = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
df_movies = pd.read_csv('rotten_tomatoes_movies.csv')
print(df_critic_reviews.info())
print(df_movies.info())

#combines the two datasets
def merge_datasets():

    df_merged = pd.merge(df_critic_reviews,df_movies,on='rotten_tomatoes_link', how='inner')
    return df_merged
    print(df_merged.info())
    
    
#chooses columns we want and filters out anything we dont want
def fix_filter_dataset(df: pd.DataFrame):

    df_fixed = df.dropna(subset=['review_content'])
    print(df_fixed.info())
    return df_fixed

def main():
    df_merged = merge_datasets()
    fix_filter_dataset(df_merged)

if __name__ == "__main__":
    main()