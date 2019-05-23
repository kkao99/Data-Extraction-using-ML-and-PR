
badaddresses = open( 'badaddresses.txt', 'r')

badaddcsv = open('badAddresses.csv', 'w')

badaddcsv.write('Address,is_address\n')

for line in badaddresses:
    line_string = "\"{}\"".format(line[:-1])
    badaddcsv.write(line_string + ",0\n")
    print(line_string)



import pandas as pd
import numpy as np


badcsv = (pd.read_csv('badAddresses.csv'))

realcsv = (pd.read_csv('realAddresses.txt'))

frames = [badcsv,realcsv]

finalDF = pd.concat(frames)

targetData = finalDF

msk = np.random.rand(len(targetData)) < 0.8

trainData = targetData[msk]
testData = targetData[~msk]


from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer()

Train = vector.fit_transform(trainData.Address)
Test = vector.transform(testData.Address)



from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(Train,trainData.is_address)

prediction = mnb.predict(Test)

testScore = np.mean(prediction == testData.is_address)

print(testScore)

