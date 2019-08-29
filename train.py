'''
Training Program
Call different models and use dataLoaders
and train them

Pass on trainLoader with fileNos and labels,
model, optimizer, learningRate, iterations,location of Train/Test folders,
batchSize, inputShape, output CLasses, rnnLayers and Size

Output returns a trained model and lossHistory over epochs
'''

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import cv2
import CRNN

def trainModel(trainLoader, model = 'CRNN', opti = 'Adam', learningRate = 0.01,
               iterations = 20, location = 'D:/AI Hack CDAC/', batchSize = 64,
               inShape = (3,129,500), outClasses = 5, rnnHiddenSize = 256,
               rnnLayers = 2):

    if model == 'CRNN':
        crnn = CRNN.CRNN(inShape,outClasses,rnnHiddenSize,rnnLayers)
        crnn = crnn.cuda()
    
    loss = nn.NLLLoss()  #If we output without softmax in model, then CrossEntropyLoss. Can softmax later for probabilites.
    if opti == 'Adam':    
        optimizer = optim.Adam(crnn.parameters(),lr=learningRate) #Need to Tune
    elif opti == 'Momentum':
        optimizer = optim.Adam(crnn.parameters(),lr=learningRate,momentum = 0.9)
    elif opti == 'Adagrad':
        optimizer = optim.Adagrad(crnn.parameters(),lr=learningRate)
    elif opti == 'RMSProp':
        optimizer = optim.RMSprop(crnn.parameters(),lr=learningRate,momentum = 0.9)
        
    lossHistory = []
    runningLoss = 0
    for epoch in range(1,iterations+1):
        crnn.train(True)
        for data in trainLoader:
        
            fileNos,labels = data[0].to(device),data[1].to(device)
            labels = labels.tolist()
            images = torch.FloatTensor(batchSize,inShape[0],inShape[1],inShape[2])
            i = -1
            for file in fileNos:
                i += 1
                fileName = location+'Train/'+str(file)+'.png'
                img = cv2.imread(fileName)
                if img is not None:
                    img = torch.from_numpy(img)
                    images[i] = img.permute(2,0,1)   # Channels, Height, Width
                else:
                    labels.pop(i)
                    fileNos.pop(i)
                    i -= 1
            torch.from_numpy(np.array(labels))
            
            optimizer.zero_grad()
            probabilities = crnn(images)
            losses = loss(labels,probabilities)
            losses.backward()
            optimizer.step()
            
            runningLoss += losses
            lossHistory.append(losses)
        if epoch % 4 == 0: 
            print('EpochNum: '+str(epoch)+' RunningLoss: '+str(runningLoss/4))
            runningLoss = 0
    
    return crnn, lossHistory