# K-Nearest-Classification
This repository includes a simple implementation of K-nearest bases digit classification from scratch. 

## Implementation details
The implementation was designed using sklearn python package.The MNIST dataset was loaded from sklearn.datasets and the k-nearest neighbour was purely implemented using Numpy python package.

## Dataset 
The dataset consisted of a total of 1797 Images of 8x8 dimension. There were of total of 10 classes (0-10) and each class consisted of ~180 Images.For the purpose of implementation ~130 images were set aside for training and the remaining for testing purpose.

## Results
It was clearly observed that as the value of K increases ,the accuracy decreases. The same pattern has been depicted through results below:
1. K=1 --> 98.4
2. K=3 --> 98.4
3. K=5 --> 97.8
4. K=7 --> 97.8
5. K=101 --> 88.9
