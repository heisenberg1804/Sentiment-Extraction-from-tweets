# Sentiment-Extraction-from-tweets
Dataset(From Kaggle Competition): Given twitter dataset has 4 columns textID, text,  sentiment, selected_text. 
Objective: To extract the relevant string from the text (tweets) according to the given sentiment of the tweet and check accuracy with the target variable i.e selected_text
My work: Performed text preprocessing and EDA on the twitter dataset which included:
          -> Cleaning the text tweets by removing punctuations, numbers, STOPWORDS, http links and eventually tokenized the text
          -> Counted the common words and unique words in the raw text for each sentiment specifically
          -> Created wordclouds , treemaps, donut, funnel and kde plots for visualization purposes
          -> Used Jaccard similarity to check the similarity between the text and selected text
Creating the model:
I treated the problem as a Named Entitiy Recognition task. I used the Spacy NER model pipeline. Treated the selected text as an Entity and custom trained the model for 30 iterations
Created 2 models for for positve and negative sentiment each. Didn't make model for neutral sentiment as from EDA it was identified that neutral text and selected text had jacc_sim ~ 1.
For evaluation i used Jaccard Score as the metric . The model achieved 0.62 and  0.60 on the training and test set respectively.
This text sequence to sequence problem can be treated as Q-A problem or regression problem also and that is the future scope of this project
