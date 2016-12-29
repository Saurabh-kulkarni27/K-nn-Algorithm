from scipy.io import mmread,mminfo
from scipy import sparse
import math
from collections import Counter
import random

#Access matrix file and convert in linked list format:
def accessDataFile(mtx_file, label_file):
    first=mmread(mtx_file)
    second=sparse.csr_matrix(first)
    dataSet=second.tolil()
    print(type(dataSet))
    labels=[]
    label_file=open(label_file)
    for line in label_file.readlines():
        n,label=line.strip().split(',')
        labels.append([n,label])
    return dataSet,labels

#Divide dataSet in Train and Test set
def splitSet(dataSet):
    trainSet={}
    testSet={}
    
    rows=random.sample(range(1838),1838)    #Random value generated in range 0-1838
    for row in rows[:int(len(rows)*0.7)]:   #trainSet is 70% of actual dataSet
        trainSet[row+1]={}
        for column in range(dataSet.shape[1]):
            if(dataSet[row,column] != 0):
                trainSet[row+1][column+1]=dataSet[row,column]
    for row in rows[int(len(rows)*0.7):]:
        testSet[row+1]={}                   #test created with 30% of actual dataset. 
        for column in range(dataSet.shape[1]):    
            if(dataSet[row,column] != 0):           
                testSet[row+1][column+1]=dataSet[row,column]
    return trainSet,testSet
        
#calculate cosine similarity
def cosine_similarity(trainSet,myTestDoc, docKey,labels,k):
    cosine={}
    for doc in trainSet.keys():
        sum_frequency=0;
        sum_frequency_train=0;
        sum_frequency_test=0
        CommonWords=(set(trainSet[doc].keys()) & set(myTestDoc.keys()))    # used set function to calculate intersection of same words between test and train documents
        for word in CommonWords:            #calculate sum of frequency of common words
            sum_frequency+=myTestDoc[word]*trainSet[doc][word]
        for word in trainSet[doc].keys():
            sum_frequency_train+=(trainSet[doc][word])**2
        for word in myTestDoc.keys():
            sum_frequency_test+=(myTestDoc[word])**2
        cosine[doc]=sum_frequency/(((sum_frequency_train)**0.5)*((sum_frequency_test)**0.5))   #cosine similarity calculation
        if cosine[doc]==1:
                        cosine[doc]=0.99999999
    
    return unweighted_classifier(cosine,labels,docKey,k),weighted_classifier(cosine, labels,docKey,k)

#for weighted K-nn classifier
def weighted_classifier(cosine, labels,docKey,k):
    votingGroup=Counter()
    for key in cosine.keys():
        cosine[key]=1/(1-cosine[key])
    cosine_weight_sort=sorted(cosine, key=cosine.get, reverse=True)
    for element in range(0,k):
        votingGroup[labels[cosine_weight_sort[element]-1][1]]+=1
    topClass=list(votingGroup)
    if (topClass[0]==labels[int(docKey)-1][1]):
        return(True)
    else:
        return(False)

#for unweighted K-nn classifier
def unweighted_classifier(cosine, labels,docKey,k):
    cosine_sort=sorted(cosine,key=cosine.get, reverse=True)
    votingGroup=Counter()
    for element in range(0,k):
        votingGroup[labels[cosine_sort[element]-1][1]]+=1
    topClass=list(votingGroup)
    if (topClass[0]==labels[int(docKey)-1][1]):
        return(True)
    else:
        return(False)

def main():
    mtx_file='news_articles.mtx'
    label_file='news_articles.labels'
    #k=int(input("Enter value of k: "))
    for k in range(1,11):
        print("For k: ",k)
        print("Step1. Accessing Data Files")
        dataSet,labels=accessDataFile(mtx_file,label_file)
        print("Step2. Spliting data set into Training and Test set")
        print("Step3. calculate Cosine Similarity ")
        print("Step4. unweighted and weighted K-nn")
        trainSet, testSet=splitSet(dataSet)
        print("Step.5 Accuracy calculation")
        unweighted_accuracy=Counter()
        weighted_accuracy=Counter()
        for docKey in testSet.keys():
            unweighted,weighted=cosine_similarity(trainSet, testSet[docKey],docKey,labels,k)
            unweighted_accuracy[unweighted]+=1
            weighted_accuracy[weighted]+=1
        print("unweighted " + str(unweighted_accuracy) + "Weighted " + str(weighted_accuracy))
main()
