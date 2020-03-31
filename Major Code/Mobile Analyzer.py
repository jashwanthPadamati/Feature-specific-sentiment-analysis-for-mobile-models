#importing necessary packages
from __future__ import absolute_import
from collections import namedtuple
import nltk
from textblob.tokenizers import word_tokenize
from textblob.decorators import requires_nltk_corpus
from textblob.base import BaseSentimentAnalyzer, DISCRETE
import pandas as pd
from pandas import DataFrame
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

#flask initialization
app=Flask(__name__, static_url_path='/')

#opening page
@app.route("/",methods=['GET','POST'])
def index():
	return(render_template('open.html'))
    
#feature extraction
def _default_feature_extractor(words):
    return dict(((word, True) for word in words))

#Naivebayes Class
class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
    kind = DISCRETE
    # Return type declaration
    RETURN_TYPE = namedtuple('Sentiment', ['classification', 'p_pos', 'p_neg'])
    #init defination
    def __init__(self, feature_extractor=_default_feature_extractor):
        super(NaiveBayesAnalyzer, self).__init__()
        self._classifier = None
        self.feature_extractor = feature_extractor
    #requires nltk corpus to train the model
    @requires_nltk_corpus
    def train(self):
        super(NaiveBayesAnalyzer, self).train()
        neg_ids = nltk.corpus.movie_reviews.fileids('neg')
        pos_ids = nltk.corpus.movie_reviews.fileids('pos')
        neg_feats = [(self.feature_extractor(
            nltk.corpus.movie_reviews.words(fileids=[f])), 'neg') for f in neg_ids]
        pos_feats = [(self.feature_extractor(
            nltk.corpus.movie_reviews.words(fileids=[f])), 'pos') for f in pos_ids]
        train_data = neg_feats + pos_feats
        self._classifier = nltk.classify.NaiveBayesClassifier.train(train_data)

    #analyze the statement
    def analyze(self, text):
        """Return the sentiment as a named tuple of the form:
        ``Sentiment(classification, p_pos, p_neg)``
        """
        # Lazily train the classifier
        super(NaiveBayesAnalyzer, self).analyze(text)
        tokens = word_tokenize(text, include_punc=False)
        filtered = (t.lower() for t in tokens if len(t) >= 3)
        feats = self.feature_extractor(filtered)
        prob_dist = self._classifier.prob_classify(feats)
        return self.RETURN_TYPE(
            classification=prob_dist.max(),
            p_pos=prob_dist.prob('pos'),
            p_neg=prob_dist.prob("neg")
        )
#when submitted
@app.route("/res",methods=['GET','POST'])	
def find():
    analyze_all=request.form.get('analyzeall')
    mobile_model=request.form.get('model_name')
    feature=request.form.get('feature')
    #get the values for the selections
    if analyze_all=='1':
        return scrapeall(feature)
        #call if all the models are to be analyzed
    else:
        return scrape(mobile_model,feature)
        #if only a model needs to be analyzed

#specifi to a phone model
def scrape(mobile_model,feature):
    #getting the dataset of csv format
    dataset=pd.read_csv('C:/Users/Admin/Desktop/Major Code/Dataset.csv')
    #filter for a model
    mobile_dataset=dataset[dataset['Model']==mobile_model]
    #construction of dataframe
    df= DataFrame(mobile_dataset,columns=['Model','Rating','Title','Review'])
    #selecting only request columns and changinf to list
    reviews=df['Review'].values.tolist()
    no_of_positive=0
    no_of_negetive=0
    #features repository 
    features={'Camera':['camera','Camera','Cam','cam','Picture','Pic','picture','pic','photo','Photo','Clarity','clarity'],
        'Battery':['battery','Battery'],
        'RAM':['RAM','Performance','performance','speed','Speed'],
        'Quality':['quality','Quality','Performance','performance','Features','features','feature','feature'],
        'Screen':['Screen','screen','Size','size','Weight','Weight'],
        'Memory':['Memory','memory','storage','Storage','Capacity','capacity']}
    #selecting feature that is selected
    val=features[feature]
    #creation of the object
    NB=NaiveBayesAnalyzer()
    #train the model
    NB.train()
    #for all the comments of the model
    for i in range(len(reviews)):
        comment=reviews[i]
        flag=0
        '''if no_of_positive+no_of_negetive>100:
            break
            --if the dataset to be limited'''
        try:
            for j in val:
                if j in comment:
                    flag=1
                    break
                    #when the word is available in comment
        except:
            continue
        if flag==0:
            continue
        anal=NB.analyze(comment)
        #check if the commeny is positive or negetive
        if anal.classification=='pos':
            no_of_positive+=1
        elif anal.classification=='neg':
            no_of_negetive+=1
    
    #validation of number of postive and negetive comments
    total_count=no_of_positive+no_of_negetive
    if total_count==0:
        no_of_negetive=1
        no_of_positive=1
    total_count=no_of_positive+no_of_negetive
    
    #percentage and rounding them off to 2 decimal points
    pos_rate=(no_of_positive/total_count)*100
    neg_rate=(no_of_negetive/total_count)*100
    pos_rate = round(pos_rate, 2)
    neg_rate = round(neg_rate, 2)
    
    #Building the graph
    labels=['Positive %','Negetive %']
    values=[pos_rate,neg_rate]
    plt.bar(labels,values,color=['green','red'])
    X_label=feature+' Feature Specific Analysis'
    plt.xlabel(X_label, fontsize=18)
    plt.ylabel('Percentage of Reviews', fontsize=18)
    plt.xticks(labels, fontsize=15, rotation=0,fontstyle='italic')
    title='Mobile Model: '+mobile_model
    plt.title(title)
   
    #concerting the graph to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    graph='data:image/png;base64,{}'.format(graph_url)
    return(render_template('output.html',graph=graph))
    	
def scrapeall(feature):
    mobile_models=['Nokia 7.1','OnePlus 7T','Samsung Galaxy M30','realme 5 Pro','OnePlus 6T','Xiaomi Mi A3','Vivo V17 Pro','OPPO F11','OnePlus 7 Pro','Redmi Note 5 Pro']
    #getting the dataset in csv form
    dataset=pd.read_csv('C:/Users/Admin/Desktop/Major Code/Dataset.csv')
    #values to be stored respectively for model
    data=[]
    data_positive=[]
    data_negetive=[]
    #object for the NaiveBayes
    NB=NaiveBayesAnalyzer()
    #train the model 
    NB.train()
    #for all the models
    for mobile_model in mobile_models:
        #choosing only for the dataset
        mobile_dataset=dataset[dataset['Model']==mobile_model]
        #Choosing the required columns
        df= DataFrame(mobile_dataset,columns=['Model','Rating','Title','Review'])
        #concert the reviews to list
        reviews=df['Review'].values.tolist()
        no_of_positive=0
        no_of_negetive=0
        #features repository 
        features={'Camera':['camera','Camera','Cam','cam','Picture','Pic','picture','pic','photo','Photo','Clarity','clarity'],
        'Battery':['battery','Battery','Charge','Charging','charge','charging'],
        'RAM':['RAM','Performance','performance','speed','Speed'],
        'Quality':['quality','Quality','Performance','performance','Features','features','feature','feature'],
        'Screen':['Screen','screen','Size','size','Weight','Weight'],
        'Memory':['Memory','memory','storage','Storage','Capacity','capacity']}
        #selecting the feature
        val=features[feature]
        #for all the reviews of the model
        for i in range(len(reviews)):
            comment=reviews[i]
            flag=0
            for j in val:
                try:
                    if j in comment:
                        flag=1
                        break
                    #if the review belong to the feature
                except:
                    continue
            if flag==0:
                continue
            #analyze the comment
            anal=NB.analyze(comment)
            #check if the comment is postive/negetive
            if anal.classification=='pos':
                no_of_positive+=1
            elif anal.classification=='neg':
                no_of_negetive+=1
             
        #calculation of total positive and negetive comments
        total_count=no_of_positive+no_of_negetive
        if total_count==0:
            no_of_negetive=1
            no_of_positive=1
        total_count=no_of_positive+no_of_negetive
    
        #rate of the positive and negetive comments rounding to 2 decimals
        pos_rate=(no_of_positive/total_count)*100
        neg_rate=(no_of_negetive/total_count)*100
        pos_rate = round(pos_rate, 2)
        neg_rate = round(neg_rate, 2)
        #append the values to the list
        data.append(mobile_model)
        data_positive.append(pos_rate)
        data_negetive.append(neg_rate)
    
    #maximum and minimum feature specific reviewed models
    maxi=data_positive[0]
    mini=data_positive[0]
    position_max=0
    position_min=0
    for i in range(1,len(data_positive)):
        if maxi<data_positive[i]:
            position_max=i
            maxi=data_positive[i]
        elif mini>data_positive[i]:
            position_min=i
            mini=data_positive[i]
    
    #building the graph 
    N = 2
    pos = (data_positive[position_max],data_positive[position_min])
    neg = (data_negetive[position_max],data_negetive[position_min])
    ind = np.arange(N)
    width = 0.40

    p1 = plt.bar(ind, pos, width)
    p2 = plt.bar(ind, neg, width,bottom=pos)
    
    plt.ylabel('Percentage')
    string='Mobile model for feature '+feature
    plt.title(string)
    plt.xticks(ind, (data[position_max],data[position_min]))
    plt.yticks(np.arange(0, 101, 10))
    plt.legend((p1[0], p2[0]), ('Positive %', 'Negetive %'))

    
    #converting the graph to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(1)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    graph='data:image/png;base64,{}'.format(graph_url)
    return(render_template('output.html',graph=graph))
    
if(__name__=="__main__"):
	app.run(debug=True)