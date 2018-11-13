import numpy as np
import os
import sys
import librosa
from sklearn.cluster import KMeans as kmeans
from sklearn.linear_model import SGDClassifier
import pandas as pd
from collections import defaultdict

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