# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:03:08 2021

@author: sahil
"""

import re
import numpy as np 
import random
import pandas as pd 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import spacy
from spacy.util import compounding
from spacy.util import minibatch
from spacy.training import Example

df = pd.read_csv('./train.csv')

#Number Of words in main Text in train set
df['Num_words_text'] = df['text'].apply(lambda x:len(str(x).split()))   
df = df[df['Num_words_text']>=3]
df_train, df_test = train_test_split(df, test_size = 0.2)

#df_test = pd.read_csv('./test.csv')
#df_submission = pd.read_csv('./sample_submission.csv')


def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


# pass model = nlp if you want to train on top of existing model 
#function to train on the data using spacy and NER
def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)

    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    #create Example object to pass in nlp.update (spacy v3 update)
                    example = Example.from_dict(doc, annotations)
                    #update model              
                    nlp.update([example],  # batch of annotations
                                drop=0.5,   # dropout - make it harder to memorise data
                                losses=losses, 
                                )
                print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')


#function to create model_output path
def get_model_out_path(sentiment):
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path



#function for creating training data 
def get_training_data(sentiment):
    '''
    Returns Training data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data


#create model for positive sentiment
sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
# For Demo Purposes I have taken 3 iterations you can train the model as you want
train(train_data, model_path, n_iter=30, model=None)


#create model for negative sentiment
sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(train_data, model_path, n_iter=3, model=None)

#Now predicting with the trained model
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_ent = [start, end, ent.label_]
        if new_ent not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text

#Now loading the model and calling the predict functions for both 'positive' and negative sentiments
selected_texts = {}
MODELS_BASE_PATH = '../working/models/'
'''
if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
        
    for index, row in df_test.iterrows():
        text = row.text
        #No model needed for Neutral sentiment as jaccard similarity
        #was very high for neutral sentiment and 
        #for text length <=2
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
        
'''
#Evaluating the predictions through JACCARD SIMILARITY
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')    
    jaccard_score_train = 0
    
    for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        
        text = row.text
        
        if row.sentiment == 'positive':
            pred = predict_entities(text, model_pos)
            selected_texts.update({row.selected_text:pred})
            jaccard_score_train += jaccard(pred, row.selected_text)
            
        elif row.sentiment == 'negative':
            pred = predict_entities(text, model_neg)
            selected_texts.update({row.selected_text:pred})
            jaccard_score_train += jaccard(pred, row.selected_text)
            
        else:
            selected_texts.update({row.selected_text:text})
            jaccard_score_train += jaccard(text, row.selected_text)           
        
print(f'Average Jaccard Score is {jaccard_score_train / df_train.shape[0]}') 

#now calculating jaccard score for test set

selected_texts_test = {}
jaccard_score_test = 0
for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
    
    text = row.text
        
    if row.sentiment == 'positive':
        pred = predict_entities(text, model_pos)
        selected_texts_test.update({row.selected_text:pred})
        jaccard_score_test += jaccard(pred, row.selected_text)
        
    elif row.sentiment == 'negative':
        pred = predict_entities(text, model_neg)
        selected_texts_test.update({row.selected_text:pred})
        jaccard_score_test += jaccard(pred, row.selected_text)
        
    else:
        selected_texts_test.update({row.selected_text:text})
        jaccard_score_test += jaccard(text, row.selected_text)           
        
print(f'Average Jaccard Score is {jaccard_score_test / df_test.shape[0]}') 


'''
#creating submission file
df_submission['selected_text'] = df_test['selected_text']
df_submission.to_csv("submission.csv", index=False)
print(df_submission.head(10))
'''      

