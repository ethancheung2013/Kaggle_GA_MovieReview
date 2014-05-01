
import pandas as pd
from textblob import TextBlob
from pandas import read_csv
from textblob.classifiers import NaiveBayesClassifier
import csv
import sys
import numpy as np
from time import time
import json
import codecs

# STEPs to clean up
# create a train1_nq.csv file by REMOVING and REPLACING all "" with nothing
# truncated first 18 reviews bc it stops there
# removed special 'tab' at 3988522
# how to deal with Mexican\English cause you can't format
t0 = time()
train_df = read_csv('train1_nq.csv', encoding='utf-8', index_col = None)

# print "start FILE PREPROCESSING to JSON ************************************************"

tr_review = np.array(train_df[:500].review.tolist(),dtype=unicode)    #np array conversion to unicode needed bc of line output in line37
tr_rating = np.array(train_df[:500].rating.tolist(),dtype=unicode)    #np.array(test_df['rating'],dtype=unicode) 

# print "start FILE PREPROCESSING to CSV ************************************************"

# tr_rating = train_df['rating'][:100]    #differences to above: no conversion to list
# tr_review = train_df['review'][:100]

newTrMerged = zip(tr_review, tr_rating)

# print "start FILE PREPROCESSING to CSV ************************************************"

print_file = False
if print_file == True:
    with open('tr_movie[:10].csv', 'wb') as outfile:
        newArry = []
        outfile.write('[')
        for review, rating in newTrMerged:
            outfile.write('("' + review.encode('ascii','ignore') + '",' + rating + '),')
        outfile.write(']')
        outfile.close()
        print("done in %fs" % (time() - t0))
        sys.exit(0)
# # print "end FILE PREPROCESSING to JSON ************************************************"

# # #out = codecs.getwriter('utf-8')(sys.stdout)
# # out = codecs.getreader('utf-8')(sys.stdout)

# # build the json train json document to NB
# # with open('tr_movie[:10].json', 'wb') as outfile:
# #     # json.dump(tr_review,tr_rating, outfile)
# #     newArry = []
# #     outfile.write('[')
# #     for review, rating in newTrMerged:
# #         # print "out.write(review) ",out.write(review)
# #         # review was upgraded from unicode to utf-8 in out.write(review) bc there were special characters found
# #         outfile.write('{"text": "' + review.encode('ascii','ignore') + '", "label": ' + rating + '},')
# #     outfile.write(']')
    
        

print "begin training"
# when files are 30MB, can you split them and pass them as arrays for learning??
# cl = NaiveBayesClassifier('tr_movie[:10].json',format='json')
# cl = NaiveBayesClassifier(train='tr_movie[:10].json', format='json')   #(newTrMerged)
#    # list of tuples
# # cl = NaiveBayesClassifier(newTrMerged)

cl = NaiveBayesClassifier(newTrMerged)
print "end training"

# # open test file and evaluate prediction probabiity
test_df = read_csv('test1_org.csv')

tr_ID = test_df['ID']#[:5]
tr_review = test_df['review']#[:5]

newTestMerged = zip(tr_review,tr_ID)

with open('result.csv', 'wb') as csvfile:
    resultwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    resultwriter.writerow(("ID","Predicted"))
    emptyCl = []
    g = (line for line in newTestMerged)
    for line in g:
        expected_label = cl.classify(line[0])
        emptyCl.append(expected_label)
        prob_dist = cl.prob_classify(line[0])
        prob_pos = prob_dist.prob("1")
        result = line[1], prob_pos 
        resultwriter.writerow(result)

print("done in %fs" % (time() - t0))



