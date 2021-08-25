# Disaster Response Pipeline Project

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

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

This project, analyzes disaster data provided by Figure Eight to build a model for an API that classifies disaster messages. The purpose is to create an app that...

## File Descriptions <a name="files"></a>

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![alt text](https://github.com/prestonb-source/disaster_pl/blob/b47e74c9161aaa484202157dcb2465400bb524c4/screenshots/graphs.JPG)
![alt text](https://github.com/prestonb-source/disaster_pl/blob/a9fc153a458372aab3a21dfac6d5df3edc173f94/screenshots/message_classifier.JPG)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

1. Udacity
2. FigureEight