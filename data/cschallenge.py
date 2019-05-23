
import random
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

realFirst = open('realNames.txt', 'r')

realLast = open('realLastName.txt', 'r')

# realNamesFinal = open("realNamesData.csv", 'w')
#
# realNamesFinal.write("Name,is_name\n")
#
# lastNameLines = realLast.readlines()
#
# for line in realFirst:
#     randomLastName = random.randint(0,len(lastNameLines)-1)
#
#     final_line = ("\"{0} {1}\"" + ",1\n").format(line[:-1], lastNameLines[randomLastName][:-1])
#
#     realNamesFinal.write(final_line)
#
# realNamesFinal.close()


letters = string.ascii_lowercase


# def createFakeName():
#
#     fakeFirstLength = random.randint(2,21)
#     fakeLastLength = random.randint(2,21)
#
#     firstName = ""
#
#     while len(firstName) != fakeFirstLength:
#
#         firstName += random.choice(letters)
#
#     lastName = ""
#
#     while(len(lastName)) != fakeLastLength:
#
#         lastName += random.choice(letters)
#
#     fullName = firstName + " " + lastName
#
#     return fullName
#
#
#
# # fakeNamesFinal = open('fakeNamesData.csv', 'w')
# #
# # fakeNamesFinal.write('Name,is_name\n')
# # for i in range(100000):
# #     name = createFakeName()
# #     fakeNamesFinal.write("\"{}\",0\n".format(name))
# # fakeNamesFinal.close()
#
# realNameDF = pd.read_csv("realNamesData.csv")
# fakeNameDF = pd.read_csv("fakeNamesData.csv")
#
# merging = [realNameDF,fakeNameDF]
#
# allNamesDF = pd.concat(merging)
#
# nameMSK = np.random.rand(len(allNamesDF)) < 0.8
#
#
# nameTrainData = allNamesDF[nameMSK]
# nameTestData = allNamesDF[~nameMSK]
#
# nameVector = TfidfVectorizer()
#
# nameVector.fit_transform(nameTrainData.Name)
# nameVector.fit(nameTestData.Name)
#
#
# NameModel = MultinomialNB()





# start of address

badaddresses = open( 'badaddresses.txt', 'r')

badaddcsv = open('badAddresses.csv', 'w')

badaddcsv.write('Address,is_address\n')

for line in badaddresses:
    line_string = "\"{}\"".format(line[:-1])
    badaddcsv.write(line_string + ",0\n")
    # print(line_string)


moreTesting = open('moreTests.txt', 'r')
moreCSV = open('more.txt', 'w')

moreCSV.write('Address,is_address\n')

for line in moreTesting:
    line_string = "\"{}\"".format(line[:-1])

    moreCSV.write(line_string + ',1\n')

moreCSV.close()


badcsv = (pd.read_csv('badAddresses.csv'))
realcsv = (pd.read_csv('more.txt'))
frames = [badcsv,realcsv]
finalDF = pd.concat(frames)

# targetData = finalDF[finalDF.is_address == 1]
targetData = finalDF

msk = np.random.rand(len(targetData)) < 0.8

trainData = targetData[msk]
testData = targetData[~msk]




vector = TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{2,}\b',ngram_range=(1,1),max_df=0.7, min_df=0.04)

Train = vector.fit_transform(trainData.Address)
Test = vector.transform(testData.Address)




mnb = MultinomialNB()
mnb.fit(Train,trainData.is_address)

svm = SGDClassifier(loss = 'hinge',alpha= 1e-3, random_state= 42)
svm.fit(Train,trainData.is_address)
svmPrediction = svm.predict(Test)
svmScore = np.mean(svmPrediction == testData.is_address)

print("svm Score \t",svmScore)


prediction = mnb.predict(Test)
testScore = np.mean(prediction == testData.is_address)
print(testScore)




finalData = pd.read_csv("prodData.txt")

prod = vector.transform(finalData.Address)



print(mnb.predict(prod))



