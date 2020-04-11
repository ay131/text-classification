import io
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos =io.open("short_reviews/positive.txt", "r",encoding="latin-1").read()
short_neg =io.open("short_reviews/negative.txt", "r",encoding="latin-1").read()

# move this up here
all_words = []
documents = []


allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

##------------------------save documents data sets  -------------------------
save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]
##------------------------save word_featureset-------------------------
save_word_features = open("pickled_algos/word_featureset.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets= [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]
##------------------------save training  and testing data-------------------------
##save testing sets
save_word_features = open("pickled_algos/testing_set.pickle", "wb")
pickle.dump(training_set, save_word_features)
save_word_features.close()
##save training sets
save_word_features = open("pickled_algos/training_set.pickle", "wb")
pickle.dump(training_set, save_word_features)
save_word_features.close()
##------------------------save classifire modles  -----------------
##save NaiveBayesClass
classifier = nltk.NaiveBayesClassifier.train(training_set)
save_classifier = open("pickled_algos/original_Naive_Bayes_Classifier.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
##save MultinomialNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
save_classifier = open("pickled_algos/Multinomial_NB_Classifier.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()
##save BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()
##save LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
save_classifier = open("pickled_algos/LogisticRegression_classifier.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()
##save LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
save_classifier = open("pickled_algos/LinearSVC_classifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

##save SGDClassifier
SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
save_classifier = open("pickled_algos/SGDC_classifier.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()