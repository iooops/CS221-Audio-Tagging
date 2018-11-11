import numpy as np
import os
import sys
import librosa
from sklearn.cluster import KMeans as kmeans
from sklearn.linear_model import SGDClassifier
import pandas as pd
from collections import defaultdict

def training_extract_progress():
    total_count=0
    for root, dirs, files in os.walk("IRMAS-TrainingData"):
        i=0
        for f in files:
                y,sr=librosa.load(librosa.util.example_audio_file(),sr=44100)
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
            if f[f.rfind('.')+1:]=='wav':
                y,sr=librosa.load(librosa.util.example_audio_file(),sr=44100)
                y_1=librosa.effects.harmonic(y)
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