# Imports
from os import terminal_size
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

# Choose 4 categories to load
categories = ['alt.atheism',
              'soc.religion.christian',
              'comp.graphics',
              'sci.med']

# Loading the 4 categories from the files
twenty_train = load_files('./ml101-tutorial/data/twenty_newsgroups/train',
                          categories=categories,
                          shuffle=True,
                          random_state=42,
                          encoding='latin1')

# Checking some properties of the files
print('Target names: ', twenty_train.target_names)
print('Length of data: ', len(twenty_train.data))
print('Length of filenames: ', len(twenty_train.filenames))

# Print the first lines of the first loaded file
print("\n".join(twenty_train.data[0].split("\n")[:3]))

# Print some target names
print(twenty_train.target_names[twenty_train.target[0]])

# Supervised learning algorithms require a category label for each document
# The category is the name of the newsgroup
print(twenty_train.target[:10])

# It is possible to get back the category names as follows:
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

# So the target attribute simply tells you which category each belongs to

## Building a Basic "Bag of Words" feature extractor

# We need to turn the text content into numerical featurevectors
#  1. Assign a fixed integer id to each word occuring in any document
#  2. For each document:
#    a. Count the number of occurrences of each word w
#    b. Store it in X[i, j]
#     . As the values of feature j
#     . Where j is the index of the word w in the dictionary (from step 1)

## Tokenizing text with scikit-learn

# Initialize a Count Vectorizer
count_vect = CountVectorizer()

# Fit the CountVectorizer to our data and transform it
X_train_counts = count_vect.fit_transform(twenty_train.data)

# Return the shape of the fit data
print('The shape of the fit data:', X_train_counts.shape)

# CountVectorizer also supports counts of N-grams of words or consecutive
#  characters.
# N-grams are runs of consecutive characters or words.
#  e.g. In the case of word bi-grams, every consecutive pair of words
#       would be a feature.

# This is the built in dictionary of feature indices
print(count_vect.vocabulary_.get(u'algorithm'))

## From occurrences to frequencies
# Occurrences are an issue, longer documents will naturally have higher
#  average count values than shorter documents.
# To avoid discrepancies, normalise the occurrences with the total word
#  count. This is called the term frequency (tf).
# Another scaling is the inverse document frequency (idf), scale down
#  words that occur commonly in other documents too.
# We combine tf and idf to get a family of weightings, tf is usually
#  multipled by idf

# Both tf and tf-idf can be computed using scikit-learn
# Initialize a tf-idf transformer and fit to the training counts
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

# Transform the counts with the transformer to get our tf for training
X_train_tf = tf_transformer.transform(X_train_counts)

# Print the shape of the training data tf
print('Shape of the tf', X_train_tf.shape)

# Rather than transforming raw counts with TfidfTransformer
#  We can use TfidfVectorizer to directly parse the dataset
tfidf_vect = TfidfVectorizer(stop_words='english',
                             max_df = 0.5,
                             min_df = 2)
X_train_tfidf = tfidf_vect.fit_transform(twenty_train.data)
print('Shape of tfidf: ', X_train_tfidf.shape)
# As seen, the number of features was reduced from 35788 to 18188.

## Exploring K-Means clustering using scikit-learn
# Now we've extracted features from our training documents
#  we can experiment with clustering.
# K-Means is one of the most intuitive clustering methods, however,
#  it has limitations.

# Initalize a KMeans clustering instance with 4 clusters
km = KMeans(4)

# Fit our instance to our training data tfidf
km.fit(X_train_tfidf)

# Print the km labels
print(km.labels_)

# The centroids of the clusters can be returned
print(km.cluster_centers_)

# Intuitively the vector that describes the centre of a cluster is just
#  like any other featurevector, every element can be interpreted
#  as the number of times a specific term occurs (of the tf-idf weight
#  of a specific term) in a hypothetical document.
# An interesting way to explore what each clutser is representing is to
#  calculate and print the top weighted terms for that cluster.
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vect.get_feature_names()
for i in range(4):
    print("Cluster {}".format(i), end="")
    for ind in order_centroids[i, :10]:
        print(' {}'.format(terms[ind], end=""))
    print()

# When we performed the clustering, we chose to use 4 clusters
#  This was intentional as we know our data comes from 4 different groups.

# A number of different metrix exist that allow us to measure how well
#  the clusters fit the known distribution of underlying newsgroups.
# One such metric is the homogeneity which is a measure of how pure the
#  clusters are with respect to the known groupings

print("Homogeneity: {}".format(metrics.homogeneity_score(twenty_train.target,
                                                         km.labels_)))
# Homogeneity scores vary between 0 and 1
#  A score of 1 indicates that the clusters match the original label
#  distribution exactly.

# Exercise: Can you print out which cluster each document belongs to?
#print(km.labels_)
#print(twenty_train.filenames)

print(km.labels_[:10])
print(twenty_train.filenames[:10])

#for doc in range(10):
#    print(twenty_train.filenames[doc], km.labels_[doc])

#print(twenty_train.filenames[km.labels_])