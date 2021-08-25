# Disaster Response Pipeline Project

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Screenshots](#screeshots)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
This code uses Python version 3.*.<br/>
The specific libraries required are: 

1. pandas
2. numpy
3. string
4. nltk
5. sklearn
6. re
7. pickle
8. time
9. joblib
10. sqlalchemy

## Project Motivation<a name="motivation"></a>

This project, analyzes disaster data provided by Figure Eight to build a model for an API that classifies disaster messages. The purpose is to create an app that identifies relevant messages which can then be directed to the most appropriate emergency agencies.

## File Descriptions <a name="files"></a>
####Data
process_data.py - ETL pipeline for data cleaning, feature extraction and loading into SQL database<br/>
disaster_categories.csv - raw data with message categories<br/>
disaster_messages.csv - raw data with messages<br/>
DisasterResponse.db - SQL database with cleaned data<br/>

###Models
train_classifier.py - machine learning pipeline to predict message categories and store model in a pkl file

###App
run.py - script to run flask app<br/>
templates - templates for flask app<br/>


## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots <a name="screenshots"></a>

1. Message Classifier:
![alt text](https://github.com/prestonb-source/disaster_pl/blob/a9fc153a458372aab3a21dfac6d5df3edc173f94/screenshots/message_classifier.JPG)

2. Message Distribution Graphs:
![alt text](https://github.com/prestonb-source/disaster_pl/blob/b47e74c9161aaa484202157dcb2465400bb524c4/screenshots/graphs.JPG)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks to Udacity for the project structure and FigureEight for the data.