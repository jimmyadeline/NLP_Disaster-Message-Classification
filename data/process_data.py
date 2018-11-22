import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from sqlalchemy import create_engine

def load_data(message_path, category_path):
    '''Load Data
    Input: two links of data path for message and category
    Output: Merged database df
    '''
    # Load messages dataset
    messages = pd.read_csv(message_path)
    # Load categories dataset
    category = pd.read_csv(category_path)
    # Merge datasets
    df = messages.merge(category, how = 'left', on = ['id'])
    df.sort_values(['id'])
    
    return df

def clean_data(df):
    ''' Clean Data: 
    - separate categories ';'
    - rename categories '-'
    - clean abnormal data
    - drop duplicate
    '''
    # Clean categories column
    category_sep = df['categories'].str.split(';', expand=True)
    
    # Generate row names
    row = category_sep.iloc[0].tolist()
    category_colnames = [i.split('-', 1)[0] for i in row]
    category_sep.columns = category_colnames
    
    # Separate 
    for col in category_sep.columns:
        category_sep[col] = category_sep[col].str.split('-', expand=True).get(1)
        category_sep[col] = pd.to_numeric(category_sep[col])
    
    # Combine two sets        
    df[category_colnames] = category_sep; df
    # df = pd.concat([df, category_sep], axis = 1)
    df.drop('categories', axis=1, inplace = True)
    
    # Clean abnormal data
    for col in category_colnames:
        if len(np.unique(df[col])) == 1:
            df.drop(col, axis=1, inplace = True)
            print('Dropped column {}'.format(col))
        elif len(np.unique(df[col])) >2:
            df = df[df[col].isin([0,1])]
        else:
            df = df  
    
    # Drop duplicate data
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    ''' Save Data: save data as DisasterResponse
    Input: cleaned df, saved file name
    Output: NA
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()