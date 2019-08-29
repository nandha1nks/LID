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

import dataLoader
import CRNN
device = torch.device("cuda")

def trainModel(trainLoader, config):
    trainLoader = dataLoader.dataloaders(config, True)

    if config['model'] == 'CRNN':
        model = CRNN.CRNN(config)
    model.to(device)
    
    loss = nn.NLLLoss()  #If we output without softmax in model, then CrossEntropyLoss. Can softmax later for probabilites.
    learningRate = config['learningRate']
    opti = config['optimizer']
    if opti == 'Adam':    
        optimizer = optim.Adam(model.parameters(),lr=learningRate) #Need to Tune
    elif opti == 'Momentum':
        optimizer = optim.Adam(model.parameters(),lr=learningRate,momentum = 0.9)
    elif opti == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(),lr=learningRate)
    elif opti == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(),lr=learningRate,momentum = 0.9)
        
    lossHistory = []
    runningLoss = 0
    print('Training Begins')
    for epoch in range(1,config['iterations']+1):
        model.train(True)
        for data in trainLoader:
        
            fileNos,labels = data
            labels = labels.tolist()
            images = torch.FloatTensor(config['batchSize'], config['inShape'][0],
                                       config['inShape'][1], config['inShape'][2])
            i = -1
            for file in fileNos:
                i += 1
                fileName = config['trainDir']+str(file)+'.png'
                img = cv2.imread(fileName)
                if img is not None:
                    img = torch.from_numpy(img)
                    images[i] = img.permute(2,0,1)   # Channels, Height, Width
                else:
                    labels.pop(i)
                    fileNos.pop(i)
                    i -= 1
            torch.from_numpy(np.array(labels))
            images.to(device)
            labels.to(device)
            
            optimizer.zero_grad()
            probabilities = model(images)
            losses = loss(labels,probabilities)
            losses.backward()
            optimizer.step()
            
            runningLoss += losses
            lossHistory.append(losses)
        if epoch % 4 == 0: 
            print('EpochNum: '+str(epoch)+' RunningLoss: '+str(runningLoss/4))
            runningLoss = 0
    print('Training Ends')
    return model, lossHistory