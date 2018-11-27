# Disaster Response Pipeline
### Build Pipelines to Classify Messages with Figure Eight

This project builds a data pipeline to prepare message data from major natural disasters around the world. I build an Adaboost pipeline to categorize emergency messages based on the needs communicated by the sender.

### Main Steps:
 - Clean data by Natural Language Processing (Normalized, tokenized and lemmatized the text messages);
 - Built up pipelines to train Adaboost and Random forest with grid search; Applied TF-IDF to assign the weights to words in the message;
 - Use website to test results. 
 
Data Source: Figure Eight, a company focused on creating datasets for AI applications, has crowdsourced the tagging and translation of messages to improve disaster relief efforts. 

### Instructions:
Run the following commands in the project's root directory to set up database and model.

- To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### Running the Web App from the Project Workspace IDE
Step 1: Type in the command line: 
    `python run.py`

Step 2: Open another Terminal Window, Type:
    `env|grep WORK`
       
Step 3: In a new web browser window, type in the following:
    `https://SPACEID-3001.SPACEDOMAIN`
where SPACEID & SPACEDOMAIN are shown in step 2.

