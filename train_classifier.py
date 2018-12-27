# Import Libraries
import sys
import pandas as pd
import numpy as np

import sqlite3
from sqlalchemy import create_engine
import os
import pickle
from tokenizer_util import tokenize

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    ''' Load data: load saved DisasterResponse 
    Input: database_filepath with dataset DisasterResponse 
    Output: X, Y, category_names(for responses)
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table ('DisasterResponse', con = engine)
    category_names = df.columns[-35:]
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X,Y,category_names


def build_model():
    ''' Model:
    - Pipeline: CountVectorizer; TfidfTransformer; AdaBoostClassifier
    - Tune Parameters
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator = AdaBoostClassifier()))
    ])
    parameters = {
        'vect__min_df':[1,10,50],
        'clf__estimator__learning_rate': [0.001, 0.01, 0.1],
        'tfidf__smooth_idf': [True, False]
    }
    model  = GridSearchCV(pipeline, param_grid = parameters, cv=2) 
    model = pipeline
    return model  

def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluate model by classification_report (sklearn.metrics)
    Input: model, X_test, Y_test, category_names
    Output: precision, recall, f1-score, support
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names, digits=2))
    
def save_model(model, model_filepath):
    ''' Save Model
    Input: model, model_filepath
    Output: pkl_file
    '''
    with open(model_filepath, 'wb') as pkl_file:
        pickle.dump(model, pkl_file)
    pkl_file.close()
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()