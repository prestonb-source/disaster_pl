# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import re
import pickle
import time
from joblib import dump

def load_data(database_filepath):
    
    """Loads data from specified filepath    
    Parameters: 
        database_filepath for sqlite database   
    Returns: 
        X (dataframe): messages
        y (dataframe): message categories (labels)
        category_names: column names for the labels   
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = 'disaster_df'
    df = pd.read_sql_table(table_name,engine)

    X = df.iloc[:,1] 
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
        
    """Processes the text data, by removing stop words and special characters, tokenizing and lemmatizing text   
    Parameters:
        text: dataframe with text to be processed
    Returns: 
        tokens: processed text    
    """ 
       
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    
    """builds model to classify messages
    Parameters:
        n/a
    Returns: 
       model: classification model
    """ 
    pipeline = Pipeline([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {'classifier__estimator__n_estimators': [10, 20, 40]}

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)   
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """Description of the function    
    Parameters:   
        model:
        X_test:
        Y_test:
        category_names:
    Returns:    
        n/a
    """
    
    y_preds = model.predict(X_test)
    print(classification_report(Y_test, y_preds, target_names=category_names))
    print('------------------------------------------------------------------')
    for i in range(Y_test.shape[1]):
        print("Category:", category_names[i],
              "\n", 
              classification_report(Y_test.iloc[:, i].values, y_preds[:, i]))
        print('Accuracy: {:,.2f}%'.format(accuracy_score(Y_test.iloc[:, i].values, y_preds[:,i])*100),"\n")
                           
def save_model(model, model_filepath):
    
    """Saves the model to the specified filepath  
    Parameters:
        model: fitted model
        model_filepath: filepath for where to save model to   
    Returns: 
        n/a   
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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