<h2 style="text-align: left;"><span style="text-decoration: underline; color: #ff6600;"><em>Sentiment-Extraction-from-<span style="color: #ff6600; text-decoration: underline;">tweets</span></em></span></h2>
<p><span style="text-decoration: underline;"><em>Dataset(From Kaggle Competition):</em></span><em> Given twitter dataset has 4 columns textID, text, sentiment, selected_text.</em></p>
<p><em><span style="text-decoration: underline;">Objective:</span> To extract the relevant string from the text (tweets) according to the given sentiment of the tweet and check accuracy with the target variable i.e selected_text</em></p>
<p><span style="text-decoration: underline;"><em>My work:</em></span><em> Performed text preprocessing and EDA on the twitter dataset which included:&nbsp;</em></p>
<ol>
<li><em>Cleaning the text tweets by removing punctuations, numbers, STOPWORDS, http links and eventually tokenized the text.</em></li>
<li><em>Counted the common words and unique words in the raw text for each sentiment specifically.</em></li>
<li><em>Created wordclouds , treemaps, donut, funnel and kde plots for visualization purposes.</em></li>
<li style="text-align: left;"><em>Used Jaccard similarity to check the similarity between the text and selected text.</em></li>
</ol>
<p><span style="text-decoration: underline;"><em>Creating the model: </em></span><em>I treated the problem as a Named Entitiy Recognition task. I used the Spacy NER model pipeline. Treated the selected text as an Entity and custom trained the model for 30 iterations.</em></p>
<p><em>Created 2 models for for positve and negative sentiment each. Didn't make model for neutral sentiment as from EDA it was identified that neutral text and selected text had jacc_sim ~ 1.</em></p>
<p><span style="text-decoration: underline;"><em>Evaluation : </em></span><em>For evaluation i used Jaccard Score as the metric . The model achieved 0.62 and 0.60 on the training and test set respectively.<br /></em></p>
<p><em>This text sequence to sequence problem can be treated as Q-A problem or regression problem also and that is the future scope of this project</em></p>
