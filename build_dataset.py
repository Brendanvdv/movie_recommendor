import pandas as pd
import numpy as np
from typing import cast

df_critic_reviews = pd.read_csv('rotten_tomatoes_critic_reviews.csv')
df_movies = pd.read_csv('rotten_tomatoes_movies.csv')

#combines the two datasets
def merge_datasets():

    df_merged = pd.merge(df_critic_reviews,df_movies,on='rotten_tomatoes_link', how='inner')
    return df_merged
    
#chooses columns we want and filters out anything we dont want
def fix_filter_dataset(df: pd.DataFrame) -> pd.DataFrame:

    wanted_columns = [
        'rotten_tomatoes_link',
        'movie_title',
        'content_rating',
        'genres',
        'original_release_date',
        'review_score',
        'review_content',
        'movie_info'
    ]

    #remove empty rows
    df_fixed = df.dropna(subset=wanted_columns)

    # take wanted columns. Use .copy() to avoid SettingWithCopyWarning
    df_filtered = df_fixed[wanted_columns].copy()

    #lowercase dataset
    for col in df_filtered.select_dtypes(include=['object']).columns:
        df_filtered.loc[:, col] = df_filtered[col].str.lower()

    #only take year from release date
    df_filtered['original_release_date'] = df_filtered['original_release_date'].str.split('-').str[0]

    #convert rating to percentage
    df_filtered['review_score'] = df_filtered['review_score'].apply(convert_rating)

    return df_filtered

def convert_rating(rating):
    
    if '/' in rating: 
        try:
            numerator, denominator = rating.split('/')
            numerator = float(numerator)
            denominator = float(denominator)
            
            percentage = round((numerator / denominator) * 100,2)

            return percentage

        except:
            return None
        
    letter_grades = {
        'a+': 97, 'a': 93, 'a-': 90,
        'b+': 87, 'b': 83, 'b-': 80,
        'c+': 77, 'c': 73, 'c-': 70,
        'd+': 67, 'd': 63, 'd-': 60,
        'f': 50
    }

    if rating in letter_grades:
        return letter_grades[rating]


def create_csv(df: pd.DataFrame):

    # df[0:30000].to_csv('final_dataset.csv')
    df.to_csv('final_dataset.csv')



def main():

    create_csv(fix_filter_dataset(merge_datasets()))

if __name__ == "__main__":
    main()