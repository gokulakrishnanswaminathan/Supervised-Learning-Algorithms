#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import csv
import copy
import math
pd.options.mode.chained_assignment = None 


# In[13]:


#reading file

trainingSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension : eg.trainingSet.csv ")),delimiter=',') 
testSet=pd.read_csv("{}".format(input("Enter the Training Dataset with csv extension :eg.testSet.csv ")),delimiter=',')   


# In[14]:


class Logistic_Regression():
    
    def __init__(self,trainingSet,testSet,stepSize,maxIterations,threshold,lamda):
        self.trainingSet=trainingSet
        self.testSet=testSet
        self.stepSize=stepSize
        self.maxIterations=maxIterations
        self.threshold=threshold
        self.lamda=lamda
    
    def sigmoid(self,z):
        estimate=1/(1+np.exp(-z))
        return estimate 
        
    def addIntercept(self,dataframe):
        dataframe.insert(loc=0,column="Intercept",value=1)
        return dataframe
    
    def labelSplit(self,dataframe,keyword):
        dataFrameFeatures=dataframe.drop(columns=keyword)
        dataFrameLabels=dataframe[keyword]
        return dataFrameFeatures,dataFrameLabels
     
    def tr(self,trainingSet,testSet):
        trainingSet=self.trainingSet
        testSet=self.testSet
        
        #Seperate Features from labels
        trainFeatures,trainLabels=self.labelSplit(trainingSet,"decision")
        testFeatures,testLabels=self.labelSplit(testSet,"decision")
      
        #Add intercept to the features
        self.addIntercept(trainFeatures)
        self.addIntercept(testFeatures)
        
        #Initialize weight's
        w=np.zeros(trainFeatures.shape[1],dtype="int")
                
        #Initialize count
        count=0
        
        while count<self.maxIterations:
            #Compute z=w^T*x_i using broadcasting technique
            w_T=np.transpose(w)
            y_hat=self.sigmoid(np.sum(w_T*trainFeatures,axis=1))
            
            #Calculating gradient
            gradient=np.sum(np.array(-trainLabels+y_hat)*np.transpose(trainFeatures),axis=1)+(self.lamda*w)
            
            #Updating weights
            w_new=w-(self.stepSize*gradient)
            diff=np.sqrt(np.sum((w_new-w)**2))
            
            if  (diff < self.threshold):
                break
            else:
                count+=1
                w=w_new
                continue
   
        trainDecision=self.sigmoid(np.sum(w*trainFeatures,axis=1))
        trainDecision[trainDecision>0.5]=1
        trainDecision[trainDecision<=0.5]=0
        
        testDecision=self.sigmoid(np.sum(w*testFeatures,axis=1))
        testDecision[testDecision>0.5]=1
        testDecision[testDecision<=0.5]=0
        
        trainResult=np.mean(trainDecision==trainLabels)*100
        testResult=np.mean(testDecision==testLabels)*100
        
        print("Training Accuracy LR: {}".format(np.round(trainResult,2)))
        print("Testing Accuracy LR: {}".format(np.round(testResult,2)))
        
        return gradient


# In[15]:


class Support_Vector_Machines():
    
    def __init__(self,trainingSet,testSet,stepSize,maxIterations,threshold,lamda):
        self.trainingSet=trainingSet
        self.testSet=testSet
        self.stepSize=stepSize
        self.maxIterations=maxIterations
        self.threshold=threshold
        self.lamda=lamda
    
    def addIntercept(self,dataframe):
        dataframe.insert(loc=0,column="Intercept",value=1)
        return dataframe 
    
    def labelSplit(self,dataframe,keyword):
        dataFrameFeatures=dataframe.drop(columns=keyword)
        dataFrameLabels=dataframe[keyword]
        return dataFrameFeatures,dataFrameLabels
       
    def svm(self,trainingSet,testSet):
        trainingSet=self.trainingSet
        testSet=self.testSet
        
        #Seperate Features from labels
        trainFeatures,trainLabels=self.labelSplit(trainingSet,"decision")
        testFeatures,testLabels=self.labelSplit(testSet,"decision")
      
        #Add intercept to the features
        self.addIntercept(trainFeatures)
        self.addIntercept(testFeatures)
        
        #Map labels to (-1,1)
        trainLabels[trainLabels==0]=-1
        testLabels[testLabels==0]=-1
        
        #Initialize weight's
        w=np.zeros(trainFeatures.shape[1],dtype="int")
                
        #Initialize count
        count=0
        
        while count<self.maxIterations:
            #Compute z=w^T*x_i using broadcasting technique
            w_T=np.transpose(w)
            
            #Broadcast to find y_estimate
            y_hat=np.sum(w_T*trainFeatures,axis=1)
            
            #Map y_estimates to [-1,1]
            y_hat[y_hat>0]=1
            y_hat[y_hat<=0]=-1
            
            hinge=y_hat*trainLabels
            hinge[hinge>=1]=0
            
            #Calculate delta:
            delta=np.sum(trainLabels*np.transpose(trainFeatures),axis=1)
 
            #Add regularization
            reg=self.lamda*w
            
            #Calculate gradient
            gradient=(1/trainFeatures.shape[0])*(reg-delta)
                 
            #Updating weights
            
            w_new=w-(self.stepSize*gradient)
            diff=np.sqrt(np.sum((w_new-w)**2))
            
            if  (diff < self.threshold):
                break
            else:
                count+=1
                w=w_new
                continue

        trainDecision=np.sum(w*trainFeatures,axis=1)
        trainDecision[trainDecision>0]=1
        trainDecision[trainDecision<=0]=-1

        testDecision=np.sum(w*testFeatures,axis=1)
        testDecision[testDecision>0]=1
        testDecision[testDecision<=0]=-1

        trainResult=np.mean(trainDecision==trainLabels)*100
        testResult=np.mean(testDecision==testLabels)*100

        print("Training Accuracy SVM: {}".format(np.round(trainResult,2)))
        print("Testing Accuracy SVM: {}".format(np.round(testResult,2)))
        
        return 


# In[16]:


classifierLm=Logistic_Regression(trainingSet,testSet,0.01,500,1*np.exp(-6),0.01)
classifierSvm=Support_Vector_Machines(trainingSet,testSet,0.01,500,1*np.exp(-6),0.01)


# In[17]:



for i in range(2):

    modelIdx=int(input("Enter model index : eg.1 or 2 "))

    if modelIdx==1:
        classifierLm.tr(trainingSet,testSet) 
    elif modelIdx==2:
        classifierSvm.svm(trainingSet,testSet)
    else:
        print("Invalid Input")

