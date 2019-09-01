'''
Testing/Validation Script

Will take in the Model location and testdirectory, testLoader as input
and find recall, precision and F1 score by inferences.
Output: Recall, Precision, F1Score Report and Confusion Matrix
'''

import torch
import CRNN
import dataLoader
import cv2
import numpy as np
from sklearn import metrics

def testModel(config):
    testLoader = dataLoader.dataLoaders(config, False)
    
    if config['cuda'] == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if config['model'] == 'CRNN':
        model = CRNN.CRNN(config)
    model.load_state_dict(torch.load(config['modelLoc'] + config['runName'] + "Weights.pth"))
    # model = torch.load(Path) for full Model Load (not prefered)
    model = model.to(device)
    
    with torch.no_grad():
        predictions = []
        groundTruths = []
        for data in testLoader:
            
            fileNos,labels = data
            labels = labels.tolist()
            fileNos = fileNos.tolist()
            imagesList = []
            i = 0
            while i < len(fileNos):
                fileName = config['testDir']+str(fileNos[i])+'.png'
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
            
            probabilities = model(images)
            pred = np.array((torch.argmax(probabilities,-1)).cpu())
            for i in range(len(labels)):
                predictions.append(pred[i])
                groundTruths.append(labels[i])
            
        confMatrix = metrics.confusion_matrix(groundTruths,predictions)
        perf = metrics.classification_report(groundTruths,predictions,digits=5)
        print('\n\nTest Performance\n\n')
        print(perf)
        print('\n\n\Confusion Matrix\n\n')
        print(confMatrix)
        return confMatrix, perf
    