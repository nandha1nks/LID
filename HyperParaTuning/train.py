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


def trainModel(config):
    trainLoader = dataLoader.dataLoaders(config, True)
    
    if config['cuda'] == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config['model'] == 'CRNN':
        model = CRNN.CRNN(config)
    model = model.to(device)
    
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
    print('\n\nTraining Begins\n\n')
    for epoch in range(1,config['iterations']+1):
        model.train(True)
        for data in trainLoader:
        
            fileNos,labels = data
            labels = labels.tolist()
            fileNos = fileNos.tolist()
            imagesList = []
            i = 0
            while i < len(fileNos):
                fileName = config['trainDir']+str(fileNos[i])+'.png'
                img = cv2.imread(fileName,0)
                if img is not None:
                    img = np.reshape(img,(config['inShape'][1],config['inShape'][2],config['inShape'][0]))
                    img = torch.from_numpy(img)
                    imagesList.append(img.permute(2,0,1))  # Channels, Height, Width
                    i += 1
                else:
                    labels.pop(i)
                    fileNos.pop(i)
            images = torch.FloatTensor(len(imagesList), config['inShape'][0],
                                       config['inShape'][1], config['inShape'][2])
            for i in range(len(imagesList)):
                images[i] = torch.from_numpy(np.array(imagesList[i]))
            images = images.to(device)
            
            label = torch.LongTensor(len(labels))
            i = -1
            for l in labels:
                i += 1
                label[i] = torch.from_numpy(np.array(l))
            labels = label.to(device)
            
            optimizer.zero_grad()
            probabilities = model(images)
            losses = loss(probabilities,labels)
            losses.backward()
            optimizer.step()
            
            runningLoss += losses.item()
            lossHistory.append(losses.item())
        if epoch % 16 == 0: 
            print('EpochNum: '+str(epoch)+' RunningLoss: '+str(runningLoss/4))
            runningLoss = 0
            
    print('\n\nTraining Ends\n\n')
    return model, lossHistory