import numpy as np
import os
import sys
import librosa
from sklearn.cluster import KMeans as kmeans
from sklearn.linear_model import SGDClassifier
from sklearn import datasets, linear_model
from collections import defaultdict
import csv
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



def training_extract():
    total_count=0
    for root, dirs, files in os.walk("IRMAS-TrainingData"):
        i=0
        for f in files:
            if not (f.startswith('.')):
                #label=f[f.find("[")+1:f.find("]")]
                #yvalues.append(label)
                y,sr=librosa.load(os.path.join(root,f),sr=44100)
                mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
                newfile=f.replace('.wav','.npy')
                new_dir='./mfcc_training'+'/'+newfile[newfile.rfind('/')+1:]
                if not os.path.exists(os.path.dirname(new_dir)):
                    os.makedirs(os.path.dirname(new_dir))
                np.save(new_dir,mfcc)
                i=i+1
                total_count+=1
                print total_count

def training_extract_progress():
    total_count=0
    for root, dirs, files in os.walk("IRMAS-TrainingData"):
        i=0
        for f in files:
            if not (f.startswith('.')):
                y,sr=librosa.load(os.path.join(root,f),sr=44100)
                y1=librosa.effects.harmonic(y)
                spec_flat=librosa.feature.spectral_flatness(y=y)
                chroma_cens=librosa.feature.chroma_cens(y=y,sr=sr)
                tonnetz=librosa.feature.tonnetz(y=y1,sr=sr)
                newfile=f.replace('.wav','.npy')
                new_dir='./spec_flat_training'+'/'+newfile[newfile.rfind('/')+1:]
                new_dir2='./chroma_sens_training'+'/'+newfile[newfile.rfind('/')+1:]
                new_dir3='./tonnetz_training'+'/'+newfile[newfile.rfind('/')+1:]
                if not os.path.exists(os.path.dirname(new_dir)):
                    os.makedirs(os.path.dirname(new_dir))
                np.save(new_dir,spec_flat)
                if not os.path.exists(os.path.dirname(new_dir2)):
                    os.makedirs(os.path.dirname(new_dir2))
                np.save(new_dir2,chroma_cens)
                if not os.path.exists(os.path.dirname(new_dir3)):
                    os.makedirs(os.path.dirname(new_dir3))
                np.save(new_dir3,tonnetz)
                i=i+1
                total_count+=1
                #print i
                print total_count

def testing_extract_progress():
    
    yvalues=[]
    for root, dirs, files in os.walk("IRMAS-TestingData-Part1"):
        i=0
        for f in files:
            if not (f.startswith('.')):
                if f[f.rfind('.')+1:]=='wav':
                    y,sr=librosa.load(os.path.join(root,f),sr=44100)
                    y1=librosa.effects.harmonic(y)
                    spec_flat=librosa.feature.spectral_flatness(y=y)
                    chroma_cens=librosa.feature.chroma_cens(y=y,sr=sr)
                    tonnetz=librosa.feature.tonnetz(y=y1,sr=sr)
                    newfile=f.replace('.wav','.npy')
                    new_dir='./spec_flat_testing'+'/'+newfile[newfile.rfind('/')+1:]
                    new_dir2='./chroma_sens_testing'+'/'+newfile[newfile.rfind('/')+1:]
                    new_dir3='./tonnetz_testing'+'/'+newfile[newfile.rfind('/')+1:]
                    if not os.path.exists(os.path.dirname(new_dir)):
                        os.makedirs(os.path.dirname(new_dir))
                    np.save(new_dir,spec_flat)
                    if not os.path.exists(os.path.dirname(new_dir2)):
                        os.makedirs(os.path.dirname(new_dir2))
                    np.save(new_dir2,chroma_cens)
                    if not os.path.exists(os.path.dirname(new_dir3)):
                        os.makedirs(os.path.dirname(new_dir3))
                    np.save(new_dir3,tonnetz)
                    i=i+1
                    print i

def test_extract():
    total_count=0
    for root, dirs, files in os.walk("IRMAS-TestingData-Part1"):
        i=0
        for f in files:
            if not (f.startswith('.')):
                if f[f.rfind('.')+1:]=='wav':
                    y,sr=librosa.load(os.path.join(root,f),sr=44100)
                    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
                    newfile=f.replace('.wav','.npy')
                    new_dir='./mfcc_testing'+'/'+newfile[newfile.rfind('/')+1:]
                    if not os.path.exists(os.path.dirname(new_dir)):
                        os.makedirs(os.path.dirname(new_dir))
                    np.save(new_dir,mfcc)
                    i+=1
                    total_count+=1
                    #print i
                    print total_count             

def build_csv_train():
     dict1={'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10}
     with open('training.csv',mode='w') as training_file:
        writer=csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header=['file name','label','tonnetz 1 mean','tonnetz 2 mean','tonnetz 3 mean',
                'tonnetz 4 mean','tonnetz 5 mean','tonnetz 6 mean','chroma sens 1 mean',
                'chroma sens 2 mean','chroma sens 3 mean','chroma sens 4 mean','chroma sens 5 mean',
                'chroma sens 6 mean','chroma sens 7 mean','chroma sens 8 mean','chroma sens 9 mean',
                'chroma sens 10 mean','chroma sens 11 mean','chroma sens 12 mean','spectral flatness mean',
                'spectral bandwidth mean','zero crossing rate mean','mfcc 1 mean','mfcc 2 mean','mfcc 3 mean',
                'mfcc 4 mean','mfcc 5 mean','mfcc 6 mean','mfcc 7 mean','mfcc 8 mean','mfcc 9 mean','mfcc 10 mean',
                'mfcc 11 mean','mfcc 12 mean','mfcc 13 mean','mfcc 14 mean','mfcc 15 mean',
                'mfcc 16 mean','mfcc 17 mean','mfcc 18 mean','mfcc 19 mean','mfcc 20 mean']
        writer.writerow(header)
        for root, dirs, files in os.walk("mfcc_training"):
            i=0
            for f in files:
                row=[]
                row.append(f.replace('.wav','.npy'))
                if not (f.startswith('.')):
                    file1=f[f.find('[')+1:f.find(']')]
                    row.append(dict1[file1])
                    tonnetz=np.load('./tonnetz_training/'+str(f)) #length=6
                    c_sens=np.load('./chroma_sens_training/'+str(f)) #length=12
                    spec_flat=np.load('./spec_flat_training/'+str(f)) #length=1
                    spec_band=np.load('./spectral_bandwidth_training/'+str(f)) #length=1
                    zero_crossing=np.load('./zero_crossing_rate_training/'+str(f)) #length=1
                    mfcc=np.load('./mfcc_training/'+str(f)) #length=20
                    row.extend(np.mean(tonnetz,axis=1))
                    row.extend(np.mean(c_sens,axis=1))
                    row.extend(np.mean(spec_flat,axis=1))
                    row.extend(np.mean(spec_band,axis=1))
                    row.extend(np.mean(zero_crossing,axis=1))
                    row.extend(np.mean(mfcc,axis=1))
                    writer.writerow(row)
                    i+=1
                    print i
                    #print len(row) #expect 43
                    #print row

def build_csv_test():
     dict1={'cel':0,'cla':1,'flu':2,'gac':3,'gel':4,'org':5,'pia':6,'sax':7,'tru':8,'vio':9,'voi':10}
     with open('test.csv',mode='w') as training_file:
        writer=csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header=['file name','label','tonnetz 1 mean','tonnetz 2 mean','tonnetz 3 mean',
                'tonnetz 4 mean','tonnetz 5 mean','tonnetz 6 mean','chroma sens 1 mean',
                'chroma sens 2 mean','chroma sens 3 mean','chroma sens 4 mean','chroma sens 5 mean',
                'chroma sens 6 mean','chroma sens 7 mean','chroma sens 8 mean','chroma sens 9 mean',
                'chroma sens 10 mean','chroma sens 11 mean','chroma sens 12 mean','spectral flatness mean',
                'spectral bandwidth mean','zero crossing rate mean','mfcc 1 mean','mfcc 2 mean','mfcc 3 mean',
                'mfcc 4 mean','mfcc 5 mean','mfcc 6 mean','mfcc 7 mean','mfcc 8 mean','mfcc 9 mean','mfcc 10 mean',
                'mfcc 11 mean','mfcc 12 mean','mfcc 13 mean','mfcc 14 mean','mfcc 15 mean',
                'mfcc 16 mean','mfcc 17 mean','mfcc 18 mean','mfcc 19 mean','mfcc 20 mean']
        writer.writerow(header)
        for root, dirs, files in os.walk("IRMAS-TestingData-Part1/Part1"):
            print "got here"
            i=0
            for f in files:
                row=[]
                if not (f.startswith('.')):
                    extension=f[f.rfind('.')+1:]
                    print extension
                    if (extension=='txt'):
                        row.append(f.replace('.txt','.npy'))
                        labels=[]
                        with open("./IRMAS-TestingData-Part1/Part1/"+str(f)) as file1:
                            for line in file1:
                                if line!="\n":
                                    label=line.strip()
                                    labels.append(dict1[label])
                        row.append(labels)
                        text=f.replace('.txt','.npy')
                        tonnetz=np.load('./tonnetz_testing/'+str(text)) #length=6
                        c_sens=np.load('./chroma_sens_testing/'+str(text)) #length=12
                        spec_flat=np.load('./spec_flat_testing/'+str(text)) #length=1
                        spec_band=np.load('./spectral_bandwidth_testing/'+str(text)) #length=1
                        zero_crossing=np.load('./zero_crossing_rate_testing/'+str(text)) #length=1
                        mfcc=np.load('./mfcc_testing/'+str(text)) #length=20
                        row.extend(np.mean(tonnetz,axis=1))
                        row.extend(np.mean(c_sens,axis=1))
                        row.extend(np.mean(spec_flat,axis=1))
                        row.extend(np.mean(spec_band,axis=1))
                        row.extend(np.mean(zero_crossing,axis=1))
                        row.extend(np.mean(mfcc,axis=1))
                        writer.writerow(row)
                        i+=1
                        print i
            
def SGD_model_predict(trainX, trainY, testX, testY, a):
    trainX=trainX.tolist()
    trainY=trainY.tolist()
    clf=SGDClassifier(verbose=0,loss='hinge',alpha=a,max_iter=1000,penalty="l2",random_state=0)
    clf.fit(trainX,trainY)
    testY_est=clf.predict(testX)
    count=0
    for i in range(len(testY)):
        if str(testY_est[i]) in testY[i]:
            count+=1
    acc=count/807.0 * 100.
    return acc

def Kmeans_model_predict(trainX,trainY):
    trainX=trainX.tolist()
    trainY=trainY.tolist()
    model=kmeans(n_clusters=11)
    labels=model.fit_predict(trainX)
    count=0
    for l in range(len(trainY)):
        if labels[l]==trainY[l]:
            count+=1
    acc=(count/float(len(trainY)))*100.
    return acc

def Kmeans_model_predict_test(testX,testY):
    testX=testX.tolist()
    testY=testY.tolist()
    model=kmeans(n_clusters=11)
    labels=model.fit_predict(testX)
    count=0
    for l in range(len(testY)):
        if str(labels[l]) in testY[l]:
            count+=1
    acc=(count/float(len(testY)))*100.
    return acc

def bagging_kfold(x_train,y_train):
    kf=KFold(n_splits=20)
    accuracy=[]
    model= BaggingClassifier()
    model2=RandomForestClassifier(n_estimators=200, max_depth=20, random_state=0)
    i=0
    for train_index,test_index in kf.split(x_train,y_train):
        xtr=x_train[train_index]
        xte=x_train[test_index]
        ytr=y_train[train_index]
        yte=y_train[test_index]
        model2.fit(xtr,ytr)
        pred=model2.predict(xte)
        accuracy.append(accuracy_score(yte,pred))
        print i
        i+=1
    print np.mean(accuracy)
        
def bagging_cross_score(x_train,y_train,x_test,y_test):
    model= BaggingClassifier()
    lasso = linear_model.Lasso()
    print cross_val_score(lasso,x_train,y_train)
    
    
if __name__ == '__main__':
    #training_extract()
    #test_extract()
    #training_extract_progress()
    #testing_extract_progress()
    #build_csv_train()
    #build_csv_test()
    
    dataset=pd.read_csv("training.csv")
    y_train=dataset['label'] 
    x_train=dataset[['tonnetz 1 mean','tonnetz 2 mean','tonnetz 3 mean',
                'tonnetz 4 mean','tonnetz 5 mean','tonnetz 6 mean','chroma sens 1 mean',
                'chroma sens 2 mean','chroma sens 3 mean','chroma sens 4 mean','chroma sens 5 mean',
                'chroma sens 6 mean','chroma sens 7 mean','chroma sens 8 mean','chroma sens 9 mean',
                'chroma sens 10 mean','chroma sens 11 mean','chroma sens 12 mean','spectral flatness mean',
                'spectral bandwidth mean','zero crossing rate mean','mfcc 1 mean','mfcc 2 mean','mfcc 3 mean',
                'mfcc 4 mean','mfcc 5 mean','mfcc 6 mean','mfcc 7 mean','mfcc 8 mean','mfcc 9 mean','mfcc 10 mean',
                'mfcc 11 mean','mfcc 12 mean','mfcc 13 mean','mfcc 14 mean','mfcc 15 mean',
                'mfcc 16 mean','mfcc 17 mean','mfcc 18 mean','mfcc 19 mean','mfcc 20 mean']].values #
    
    dataset2=pd.read_csv("test.csv")
    y_test=dataset2['label']
    x_test=dataset2[['tonnetz 1 mean','tonnetz 2 mean','tonnetz 3 mean',
                'tonnetz 4 mean','tonnetz 5 mean','tonnetz 6 mean','chroma sens 1 mean',
                'chroma sens 2 mean','chroma sens 3 mean','chroma sens 4 mean','chroma sens 5 mean',
                'chroma sens 6 mean','chroma sens 7 mean','chroma sens 8 mean','chroma sens 9 mean',
                'chroma sens 10 mean','chroma sens 11 mean','chroma sens 12 mean','spectral flatness mean',
                'spectral bandwidth mean','zero crossing rate mean','mfcc 1 mean','mfcc 2 mean','mfcc 3 mean',
                'mfcc 4 mean','mfcc 5 mean','mfcc 6 mean','mfcc 7 mean','mfcc 8 mean','mfcc 9 mean','mfcc 10 mean',
                'mfcc 11 mean','mfcc 12 mean','mfcc 13 mean','mfcc 14 mean','mfcc 15 mean',
                'mfcc 16 mean','mfcc 17 mean','mfcc 18 mean','mfcc 19 mean','mfcc 20 mean']].values
    
    bagging_kfold(x_train,y_train)
    bagging_kfold(x_test,y_test)
    #bagging_cross_score(x_train,y_train,x_test,y_test)
    
    #print "Time to test model"
    #alphas=[0.0001,0.001,.01,0.1,1,10]
    #alphas=[100,1000,10000]
    #valid_acc=[]
    #for a in alphas:
    #    acc=model_predict(x_train,y_train,x_test,y_test,a)
    #    valid_acc.append(acc)
    #print valid_acc
    #print acc
    
    