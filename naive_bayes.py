import os
import numpy as np
from collections import defaultdict
np.seterr(divide = 'ignore')
from collections import Counter

class NaiveBayes():

    def __init__(self):
        self.class_dict = {0: 'neg', 1: 'pos'}
        self.feature_dict = defaultdict()
        self.prior = None
        self.likelihood = None
        self.bigdoc = []
        self.feature_vec = None
        self.features = None
        self.inv_feature_dict = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    '''

    def train(self, train_set):
        # iterate over training documents
        neglst = []
        poslst = []
        negfile = []
        posfile = []

        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    file_list = [line.rstrip('\n') for line in f]
                    str1 = ''.join(str(e) for e in file_list)
                    if root[-3:] == self.class_dict[0]:
                        neglst.append(name)
                        negfile.extend(str1.split())
                    elif root[-3:] == self.class_dict[1]:
                        poslst.append(name)
                        posfile.extend(str1.split())

        # calculate the vocabulary
        voca = set(negfile + posfile)
        with open("negative-words.txt") as file:
            neg_senti_word = file.read().split()

        with open("positive-words.txt") as file:
            pos_senti_word = file.read().split()

        senti_words = neg_senti_word + pos_senti_word
        self.features = set([word for word in senti_words if word in voca])

        for index, words in enumerate(self.features):
            self.feature_dict[index] = words

        # added(created an inverse feature dictionary)
        self.inv_feature_dict = {value: key for key, value in self.feature_dict.items()}

        # calculate the num of negative list and positive list
        negnum = len(neglst)
        posnum = len(poslst)
        num_doc = len(neglst) + len(poslst)
        p1 = np.log(negnum / num_doc)
        p2 = np.log(posnum / num_doc)
        self.prior = np.array([p1, p2])

        self.bigdoc = {'neg': negfile, 'pos': posfile}

        self.likelihood = np.zeros(shape=(len(self.class_dict), len(self.feature_dict)))

        # Calculate the total freq of occurrences of w in neg and pos
        pos_word_counter = defaultdict(int)
        neg_word_counter = defaultdict(int)
        for word in self.bigdoc['pos']:
            pos_word_counter[word] += 1
        for word in self.bigdoc['neg']:
            neg_word_counter[word] += 1

        negcount = 0
        poscount = 0
        # Calculate the tokens in negfile and posfile
        for negitem in negfile:
            negcount += 1

        for positem in posfile:
            poscount += 1

        # added(get the likelihood )
        vocab_length = len(voca)
        for feature in self.features:
            self.likelihood[0][self.inv_feature_dict[feature]] = np.log((neg_word_counter[feature] + 1) / (negcount + vocab_length))
            self.likelihood[1][self.inv_feature_dict[feature]] = np.log((pos_word_counter[feature] + 1 ) / (poscount + vocab_length))

        return self.prior
        return self.likelihood

    #Tests the classifier on a development or test set.
    #Returns a dictionary of filenames mapped to their correct and predicted
    #classes such that:
    #results[filename]['correct'] = correct class
    #results[filename]['predicted'] = predicted class


    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    filelst = [line.strip('\n') for line in f]
                    str2 = ''.join(str(e) for e in filelst)
                    lst2 = str2.split(" ")
                    feature_word_dict = defaultdict(int)
                    # calculate how many times the feature word appears in the document
                    for w in self.inv_feature_dict:
                        if w in lst2:
                            for item in lst2:
                                if w == item:
                                    feature_word_dict[w] += 1
                        else:
                            feature_word_dict[w] = 0

                    self.feature_vec = [feature_word_dict.get(w, 0) for w in self.inv_feature_dict.keys()]

                    # add self.prior to this vector
                    sum = self.prior + np.dot(self.likelihood, self.feature_vec)
                    if root[-3:] == self.class_dict[0]:
                        results[name]['correct'] = 'neg'
                    elif root[-3:] == self.class_dict[1]:
                        results[name]['correct'] = 'pos'

                    results[name]['predicted'] = np.argmax(sum, axis=None)
        # get most likely class
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''

    def evaluate(self, results):
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        for key in results:
            if results[key]['correct'] == 'neg':
                if results[key]['predicted'] == 0:
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[1][0] += 1
            else:
                if results[key]['predicted'] == 1:
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[0][1] += 1

        # for class_1
        precision1 = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        recall1 = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
        f1score1 = (2*precision1*recall1)/(precision1+recall1)

        #for class_2
        precision2 = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
        recall2 = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
        f1score2 = (2*precision2*recall2)/(precision2+recall2)

        accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[0][0]+confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])

        print("This is the precision for class 1")
        print(precision1)
        print("This is the recall for class 1")
        print(recall1)
        print("This is the F1-Score for class 1")
        print(f1score1)
        print()
        print("This is the precision for class 2")
        print(precision2)
        print("This is the recall for class 2")
        print(recall2)
        print("This is the F1-Score for class 2")
        print(f1score2)
        print()
        print("This is the overall accuracy")
        print(accuracy)


if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    results = nb.test('movie_reviews/dev')
    nb.evaluate(results)