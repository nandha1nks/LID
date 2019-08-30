# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:43:02 2019

@author: KETAN
"""

import os
import librosa
import numpy as np
from pydub import AudioSegment as As
dur=5*60

length,audioo,label=[],[],[]
address=r"/home/aih04/dataset"
os.chdir(address)
n=np.zeros((4))
f=0
for root, subdirs, files in os.walk('.'):
    print(root, 'lol',subdirs,'::')
    if subdirs!=[]:
        f+=1
        print(f,'ketan')
    for file in files:
        address=root+'\\'+file
        if (os.path.splitext(file)[1]=='.wav')&(file[0]!='.'):
            audio,fs=librosa.load(address,sr=16000,duration=dur)
            save_address=r'/home/aih04/dataset'+str(f-1)+'/'+str(f-1)+'.'+str(int(n[f-2]))
            np.save(save_address,audio)   
            
            label.append(f-2)
            le=len(audio)/fs/60     
           
            n[f-2]+=1
            print(n[f-2],':',file)
        
        
        
