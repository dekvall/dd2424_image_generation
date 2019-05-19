import tensorflow as tf
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import scipy.io as sc
sns.set()
#import IPython.display

data_train_path='./dataset/training/'
data_test_path='./dataset/test/'
batchsize=100
SAMPLINGRATE=16000
PADDING=44
MFCCSNR=20
labels_map={'bed':1,'bird':2,'cat':3,'dog':4,'down':5,'eight':6,'five':7,'four':8,'go':9,'happy':10,'house':11,'left':12,'marvin':13,
'nine':14,'no':15,'off':16,'on':17,'one':18,'right':19,'seven':20,'sheila':21,'six':22,'stop':23,'three':24,'tree':25,'two':26,'up':27,'wow':28,'yes':29,'zero':30}

def one_hot_vector(label):
    encoding=[0]*len(labels_map)
    encoding[labels_map[label]-1]=1
    return encoding

def get_mfcc(path, padding=PADDING):
    wave,sr=librosa.load(path,sr=SAMPLINGRATE,mono=True)
    #librosa.display.waveplot(wave,sr=sr)
    mfccs=librosa.feature.mfcc(y=wave,sr=SAMPLINGRATE,n_mfcc=MFCCSNR)
    mfccs=np.pad(mfccs,((0,0),(0,padding-len(mfccs[0]))),mode='constant')
    return mfccs

def load_batch(size,path):
    X=[] #mfccs for each audio
    Y=[] #one hot vectors of labels
    Z=[] #labelse
    path=os.path.join(path, '*','*.wav')
    waves=tf.gfile.Glob(path) #all files in subdirectories
    allFiles=len(waves)
    print("Number of files being processed: ",allFiles)
    #waves=waves[0:size]
    for wave_path in waves:
        _,label=os.path.split(os.path.dirname(wave_path))
        X.append(get_mfcc(wave_path))
        Y.append(one_hot_vector(label))
        Z.append(labels_map[label])
    return X,Y,Z

def load_data(mode):
    if mode=='train':
        path=data_train_path
        saveFile='train20.mat'
    elif mode=='test':
        path=data_test_path
        saveFile='test20.mat'
    else:
        saveFile=''
    print("Processing data in ", path)
    X,Y,Z=load_batch(10,path)
    arrayformx=np.array(X)
    arrayformy=np.array(Y)
    try:
        sc.savemat(saveFile, dict(mfccs=arrayformx, onehot=arrayformy, labels=Z))
    except:
        sc.savemat('some20.mat', dict(mfccs=arrayformx, onehot=arrayformy, labels=Z))
    print("Saved processed data in", saveFile)


load_data('train')
