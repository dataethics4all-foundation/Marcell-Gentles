

import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import contractions
from gensim.models import Word2Vec
import os

#pass in a path to a csv
def clean(rawData, number= 500):


    halfOfCorpusSize = 800000

    ###

    class Tweet(object):
        #pass in a Pandas df
        def __init__(self, data):
            self.label = data["Label"]
            self.text = data["Text"]

        def clean(self):
            edit = self.text.lower()
            edit = re.sub("@[A-Za-z0-9_]+","", edit)
        ### edit = re.sub("#","", edit)
            edit = re.sub(r"http\S+", "", edit)
            edit = re.sub(r"www.\S+", "", edit)    
            edit = re.sub("[^a-z0-9'’#]"," ", edit)

            expanded_words = []
            for word in edit.split():
                expanded_words.append(contractions.fix(word))
            edit = ' '.join(expanded_words)

    ### splitting, joining, and splitting again is necessary because the contraction fixer
    ### stores both resulting words in one token.

            edit = edit.split()

            sw = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
                  'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been',
                  'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
                  'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
                  'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
                  'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
                  'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself',
                  'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've",
                  'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's",
                  'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of',
                  'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
                  'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd",
                  "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than',
                  'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
                  'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
                  "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
                  'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
                  "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which',
                  'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would',
                  "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
                  'yourself', 'yourselves']

            edit = [w for w in edit if not w in sw]
            self.tokenized = edit

            stemmed = [ps.stem(i) for i in edit]

            return " ".join(stemmed)

    ###

######### testing
    df1 = pd.read_csv(rawData, names=["Label", "Text"], usecols=[0,5], nrows= number, encoding= "latin")
    df2 = pd.read_csv(rawData, names=["Label", "Text"], usecols=[0,5], nrows= number, encoding= "latin", skiprows= halfOfCorpusSize)
    import_df = pd.concat([df1, df2])
#########



    #import_df = pd.read_csv(rawData, names=["Label", "Text"], usecols=[0,5], nrows= number)

    tweets = []
    df = pd.DataFrame(tweets, columns=["Label", "Text", "Cleaned"])


    for i in range(len(import_df)):
        tweets.append(Tweet(import_df.iloc[i]))
        new_df = pd.DataFrame([[tweets[i].label, tweets[i].text, tweets[i].clean()]], \
                              columns=["Label", "Text", "Cleaned"])
        df = pd.concat([df, new_df], axis=0)
    df.reset_index(inplace=True)
    return df

    ###

    # I need to find out why I need to add parentheses to the tweet.preprocess part (otherwise bound error)

def CleanDF(inputDF):


    
    class Tweet(object):
        #pass in a Pandas df
        def __init__(self, data):
            self.id = data["ID"]
            self.text = data["Text"]

        def clean(self):
            edit = self.text.lower()
            edit = re.sub("@[A-Za-z0-9_]+","", edit)
        ### edit = re.sub("#","", edit)
            edit = re.sub(r"http\S+", "", edit)
            edit = re.sub(r"www.\S+", "", edit)    
            edit = re.sub("[^a-z0-9'’#]"," ", edit)

            expanded_words = []
            for word in edit.split():
                expanded_words.append(contractions.fix(word))
            edit = ' '.join(expanded_words)

    ### splitting, joining, and splitting again is necessary because the contraction fixer
    ### stores both resulting words in one token.

            edit = edit.split()

            sw = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
                  'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been',
                  'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
                  'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
                  'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
                  'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
                  'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself',
                  'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've",
                  'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's",
                  'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of',
                  'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
                  'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd",
                  "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than',
                  'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
                  'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
                  "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
                  'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
                  "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which',
                  'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would',
                  "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
                  'yourself', 'yourselves']

            edit = [w for w in edit if not w in sw]
            self.tokenized = edit

            stemmed = [ps.stem(i) for i in edit]

            return " ".join(stemmed)

    ###


    import_df = pd.DataFrame(inputDF, columns= ['ID', 'Text'])
#########



    #import_df = pd.read_csv(rawData, names=["Label", "Text"], usecols=[0,5], nrows= number)

    tweets = []
    df = pd.DataFrame(tweets, columns=["ID", "Text", "Cleaned"])


    for i in range(len(import_df)):
        tweets.append(Tweet(import_df.iloc[i]))
        new_df = pd.DataFrame([[tweets[i].id, tweets[i].text, tweets[i].clean()]], \
                              columns=["ID", "Text", "Cleaned"])
        df = pd.concat([df, new_df], axis=0)
    df.reset_index(inplace=True)
    return df


def vectorize(cleanedData, vocab= None, makeCSV= False):
    bagv = CountVectorizer(vocabulary= vocab if vocab != None else None)
    bag = bagv.fit_transform(cleanedData["Cleaned"]) if vocab== None else bagv.transform(cleanedData["Cleaned"])
    bagVocab = bagv.vocabulary_


    tfidfv = TfidfVectorizer(vocabulary= vocab if vocab != None else None)
    tfidf = tfidfv.fit_transform(cleanedData["Cleaned"]) if vocab== None else tfidfv.transform(cleanedData["Cleaned"])
    tfidfVocab = tfidfv.vocabulary_

    column_list = []
    for word in bagVocab:
        column_list.append("B: " + word)
    bag_dense_df = pd.DataFrame(bag.todense(), columns = column_list)
    bag_dense_df.reset_index(inplace=True)

    column_list = []
    for word in tfidfVocab:
        column_list.append("T: " + word)
    tfidf_dense_df = pd.DataFrame(tfidf.todense(), columns = column_list)
    tfidf_dense_df.reset_index(inplace=True)

    cleanedData = cleanedData.drop('index', axis=1)

    ###

    if makeCSV == True:
        cleanedData.to_csv("cleaned_data.csv")

    vocab = {'BoW': bagVocab, 'TFIDF': tfidfVocab}
    
    return cleanedData, vocab

print(vectorize(clean("training.1600000.processed.noemoticon.csv", 5)))
