#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )


# In[11]:


# Verify the number of reviews that were read (100,000 in total)
haha = train["review"].size; hihi = test["review"].size
hoho = unlabeled_train["review"].size
print(f"Read {haha} labeled train reviews, {hihi} labeled test reviews, and {hoho} unlabeled reviews\n")
#print(f"Read {train["review"].size} labeled train reviews, {hihi} labeled test reviews, and {hoho} unlabeled reviews\n")


# In[12]:


# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


# In[13]:


# Download the punkt tokenizer for sentence splitting
import nltk.data
#No need to run the below line again after it has been run 
#nltk.download()
nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[14]:


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence,               remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[15]:


sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


# In[16]:


print(len(sentences))


# In[17]:


print(sentences[0])


# In[18]:


print(sentences[1])


# In[19]:


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[20]:


model.doesnt_match("man woman child kitchen".split())


# In[22]:


model.doesnt_match("paris berlin london austria".split())


# In[23]:


model.most_similar("man")


# In[24]:


model.most_similar("asian")


# In[25]:


model.most_similar("awful")


# In[26]:


haha = model.wv['awful']
haha, haha.size


# In[27]:


#part 3 started here
#changes below were made according to 
#https://groups.google.com/g/gensim/c/hlYgjqEVocw
#some methods and attributes were moved to
#KeyedVectors class from word2vec class
from gensim.models import Word2Vec, KeyedVectors


# In[28]:


model = KeyedVectors.load("300features_40minwords_10context")


# In[29]:


#changed: wv. added
type(model.wv.syn0)


# In[30]:


model.wv.syn0.shape

model.wv['flower']
# In[37]:


import gensim
all_stopwords = set(gensim.parsing.preprocessing.STOPWORDS)
all_stopwords


# In[43]:


#be careful: nword and counter must be integers --Chiu
import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    index2word_set2 = all_stopwords
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set and word in index2word_set2: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
        if counter%1000 == 0:
            haha = counter; hihi = len(reviews)
            print(f"Review {haha} of {hihi}") #% (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
        #reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs


# In[51]:


index2word_set = set(model.wv.index2word)
index2word_set2 = all_stopwords

haha = []
for hihi in index2word_set2:
    if hihi not in index2word_set:
        haha.append(hihi)
print(haha), print(index2word_set2)


# In[44]:


type(model)


# In[45]:


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_reviews = []
for review in train["review"]:
    #clean_train_reviews.append( review_to_wordlist( review, \
        #remove_stopwords=True ))
    clean_train_reviews.append( review_to_wordlist( review ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    #clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    clean_test_reviews.append( review_to_wordlist( review ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )


# In[46]:


# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )


# In[47]:


print(output)


# In[51]:


from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = int(num_clusters) )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")


# In[52]:


# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number                                                                                            
word_centroid_map = dict(zip( model.wv.index2word, idx ))


# In[53]:


word_centroid_map, len(word_centroid_map.keys()), 


# In[54]:


#key, val = word_centroid_map.items()[2]
type(word_centroid_map)


# In[55]:


# For the first 10 clusters
for cluster in range(0,10):
    #
    # Print the cluster number  
    #print "\nCluster %d" #% cluster
    print(f"\nCluster {cluster}")
    #
    # Find all of the words for that cluster number, and print them out
    a_view = word_centroid_map.items()
    tuples = list(a_view)
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if( tuples[i][1] == cluster ):
            words.append(tuples[i][0])
    print(words)


# In[56]:


def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# In[57]:


# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["review"].size, int(num_clusters)),     dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review,         word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros((test["review"].size, int(num_clusters)),     dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review,         word_centroid_map )
    counter += 1


# In[58]:


# This cell take some minutes
# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




