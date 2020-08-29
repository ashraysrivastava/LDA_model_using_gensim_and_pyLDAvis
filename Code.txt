#importing packages
import spacy
spacy.load('en_core_web_lg')
from spacy.lang.en import English
import nltk
parser = English()
import random
from gensim import corpora
import gensim
import pyLDAvis.gensim
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def tokenize(text):#tokenize the data and remove noises and convert everything to lower letters
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# from nltk.corpus import wordnet as wn
# def get_lemma(word):
#     lemma = wn.morphy(word)
#     if lemma is None:
#         return word
#     else:
#         return lemma
    

def get_lemma2(word):#doing text lemmization from wordnet in nltk
    return WordNetLemmatizer().lemmatize(word)

#nltk.download('stopwords')#to download stopwords
en_stop = set(nltk.corpus.stopwords.words('english'))
noise_list=["-",'.','req/','/',')','(','prb\\','prb/',',','//','>','\\','q/','"','#','?']
def prepare_text_for_lda(text):#final method to prepare text for lda model
    tokens = tokenize(text)
#     #tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma2(token) for token in tokens]
    tokens = [token for token in tokens if token not in noise_list]
#     print(tokens)
    return tokens

#opening the file
text_data = []
with open('D://ZZ Ashray(698306)//remedytickets//outlook1.csv') as f:
    for line in f:
        #tokens=entity_extraction(line)
        #print(tokens)
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
        #print(tokens)
            text_data.append(tokens)
            #print(text_data)
            
#creating dictionary and corpus for LDA model
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


#creating and training the model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           alpha='auto',
                                           per_word_topics=True)
ldamodel.save('model.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
#displaying the model
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model.gensim')
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)



