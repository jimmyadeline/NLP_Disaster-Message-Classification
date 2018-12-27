# Disaster Response Pipeline
### Build Pipelines to Classify Messages with Figure Eight

This project builds a data pipeline to prepare message data from major natural disasters around the world. I build an Adaboost pipeline to categorize emergency messages based on the needs communicated by the sender.

#### Main Steps:
 - Clean data by Natural Language Processing (Normalized, tokenized and lemmatized the text messages);
 - Built up pipelines to train Adaboost and Random forest with grid search; Applied TF-IDF to assign the weights to words in the message;
 - Use Adaboost in the final model and test results in the website. 
 
Data Source: Figure Eight, San Francisco, CA

#### Install the dependencies
If you are running this in your local environment, run
`conda install --file requirements.txt`  
Or
`pip install -r requirements.txt`
to install the required python module dependencies

#### Instructions: 
Run the following commands in the project's root directory to set up database and model.

- To run ETL pipeline that cleans data and stores in database

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- To run ML pipeline that trains classifier and saves
    
    `python train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### Running the Web App from the Project Workspace IDE
Step 1: Type in the command line: 
    `python run.py`

Step 2: Open another Terminal Window, Type:
    `env|grep WORK`
       
Step 3: In a new web browser window, type in the following:
    `https://SPACEID-3001.SPACEDOMAIN`
where SPACEID & SPACEDOMAIN are shown in step 2.


#### How to deploy it to heroku
You can refer to this tutorial : https://github.com/datademofun/heroku-basic-flask

You might also run into the following error, when deploying to Heroku : `AttributeError: module '__main__' has no attribute 'tokenize'`, 
check out this post : https://stackoverflow.com/q/53936736/2943842  to make sure tokenize()
 function is properly imported on both `train_classifier.py` and `run.py`
