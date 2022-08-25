from sklearn.datasets import load_digits
from sklearn.utils import shuffle
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from statistics import mode
import itertools

def majority_class(dist):
    classes=[]
    
    for i in range(len(dist)):
        classes.append(dist[i][0])
    print(classes,"\t",mode(classes))
    return mode(classes) #return the mode of the list to get the majority of the vote

#now test each data in the test set using nearest neighbour classifier
def K_nearest(trainx,trainy,testx,testy,K):
    #initialize the counter for correct and wrong classification
    correct=0
    wrong=0
    #convert each and every list to numpy array to make computations faster
    trainx=np.array(trainx) 
    trainy=np.array(trainy)
    testx=np.array(testx)
    testy=np.array(testy)
    #now compute the distance between the test point and all the points in the training data
    dist=[]

    for i in range(testx.shape[0]):
        for j in range(trainx.shape[0]):
            dist.append((trainy[j],np.linalg.norm(testx[i] - trainx[j]))) #computing euclidean norm using np.linalg
        # sorted_dist=sorted(dist.items(), key=lambda item: item[1])
        dist.sort(key = lambda x: x[1]) #sort the list as per the distance
        print(dist[:K])
        if majority_class(dist[:K])==testy[i]: #consider the first k nearest neighbours to classify the test point
            print("correct")
            correct+=1
        else:
            print("wrong")
            wrong+=1
        dist=[]
        sorted_dist=[]
    return correct,wrong


np.set_printoptions(threshold=sys.maxsize) #to print the entire numpy array

data,target = load_digits(return_X_y=True) #load the entire dataset into data and target variables 
data,target = shuffle(data,target,random_state=40)

#adding 50 images of each class to the test set
testx=[]
testy=[]
trainx=[]
trainy=[]
count=0
for i in range(10):
    for j in range(target.shape[0]):
        if target[j]==i and count<50: #untill count 50 keep adding the specific class image to testx after which add them to trainx
            testx.append(data[j])
            testy.append(target[j])
            count+=1
        if target[j]==i and count>=50:
            trainx.append(data[j])
            trainy.append(target[j])
            count+=1
    count=0


print(len(trainx),len(testx))

correct,wrong=K_nearest(trainx,trainy,testx,testy,101) #set K as per the requirement
print(correct,"__",wrong)
accuracy=correct/(correct+wrong)*100
print(accuracy)

#K=1, accuracy=98.4
#K=3, accuracy=98.4
#K=5, accuracy=97.8
#K=7, accuracy=97.8










