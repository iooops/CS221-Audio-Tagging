import numpy as np
import os
import sys
import librosa
from sklearn.linear_model import SGDClassifier
import pandas as pd
from collections import defaultdict


def training_extract():
    yvalues=[]
    for root, dirs, files in os.walk("IRMAS-TrainingData"):
        i=0
        for f in files:
                label=f[f.find("[")+1:f.find("]")]
                yvalues.append(label)
                y,sr=librosa.load(librosa.util.example_audio_file(),sr=44100)
                mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=100)
                newfile=f.replace('.wav','.npy')
                new_dir='./mfcc_training'+'/'+newfile[newfile.rfind('/')+1:]
                if not os.path.exists(os.path.dirname(new_dir)):
                    os.makedirs(os.path.dirname(new_dir))
                np.save(new_dir,mfcc)
                i=i+1
                print i
    return yvalues

def test_extract():
    for root, dirs, files in os.walk("IRMAS-TestingData-Part1"):
        i=0
        for f in files:
            if f[f.rfind('.')+1:]=='wav':
                y,sr=librosa.load(librosa.util.example_audio_file(),sr=44100)
                mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=100)
                newfile=f.replace('.wav','.npy')
                new_dir='./mfcc_testing'+'/'+newfile[newfile.rfind('/')+1:]
                if not os.path.exists(os.path.dirname(new_dir)):
                    os.makedirs(os.path.dirname(new_dir))
                np.save(new_dir,mfcc)
                i=i+1
                print i


def scale_and_get_y_train():
    dict1={'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10}
    mfcc_mat=np.zeros(shape=(100,6705))
    y_training=[]
    for root, dirs, files in os.walk("mfcc_training"):
        i=0
        for f in files:
            if not (f.startswith('.')):
                #text=f.replace('.npy','.txt')
                file1=f[f.find('[')+1:f.find(']')]
                y_training.append(dict1[file1])
                mfcc=np.load('./mfcc_training/'+str(f))
                mean_temp=np.mean(mfcc, axis=1)
                mfcc_mat[:,i]=np.mean(mfcc,axis=1)
                print i
                i=i+1
    return mfcc_mat,y_training

def scale_and_get_y_test():
    dict1={'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10}
    mfcc_mat=np.zeros(shape=(100,807))
    y_test=[]
    i=0
    for root,dirs,files in os.walk("mfcc_testing"):
        for f in files:
            extension=f[f.rfind('.')+1:]
            if (extension=='npy'):
                text=f.replace('.npy','.txt')
                labels=[]
                with open("./mfcc_testing/"+str(text)) as file1:
                    for line in file1:
                        if line!="\n":
                            label=line.strip()
                            labels.append(dict1[label])
                y_test.append(labels)
                mfcc=np.load("./mfcc_testing/"+str(f))
                mean_temp=np.mean(mfcc, axis=1)
                mfcc_mat[:,i]=np.mean(mfcc,axis=1)
                print i
                i=i+1
    return mfcc_mat,y_test
            
def model_predict(trainX, trainY, testX, testY, a):
    clf=SGDClassifier(verbose=0,loss='hinge',alpha=a,max_iter=1000,penalty="l2",random_state=0)
    clf.fit(trainX,trainY)
    testY_est=clf.predict(testX)
    count=0
    for i in range(len(testY)):
        if testY_est[i] in testY[i]:
            count+=1
    acc=count/807.0 * 100.
    return acc
    
if __name__ == '__main__':
    #yvalues=training_extract()
    #test_extract()
    
    print "Getting training set"
    trainX,trainY=scale_and_get_y_train()
    print "Getting testing set"
    testX,testY=scale_and_get_y_test()

    print "Normalizing features"
    trainX=trainX.T
    trainX_mean=np.mean(trainX,axis=0)
    trainX=trainX-trainX_mean
    trainX_std=np.std(trainX,axis=0)
    trainX=trainX/(trainX_std+1e-5)
    
    testX=testX.T
    testX=testX-trainX_mean
    testX=testX/(trainX_std+1e-5)
    
    print "Time to test model"
    #alphas=[0.0001,0.001,.01,0.1,1,10]
    alphas=[10,100,1000,10000]
    valid_acc=[]
    for a in alphas:
        acc=model_predict(trainX,trainY,testX,testY,a)
        valid_acc.append(acc)
    print valid_acc
    
    