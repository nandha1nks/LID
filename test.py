'''
Testing/Validation Script

Will take in the Model location and testdirectory, testLoader as input
and find accuracy, recall, precision and F1 score by inferences.
Output: ACC, REC, PRE, F1S for indiv classes and together
'''

import torch
import CRNN
import dataLoader
import cv2
import numpy as np

device = torch.device("cuda")

def evaluate(config):
    testLoader = dataLoader.dataloaders(config, False)
    
    if config['model'] == 'CRNN':
        model = CRNN.CRNN(config)
    model.load_state_dict(torch.load(config['modelLoc'] + config['runName'] + "Infer.pth"))
    # model = torch.load(Path) for full Model Load (not prefered)
    model.to(device)
    
    with torch.no_grad():
        total = np.zeros((config['outClasses'],1))
        TP = np.zeros((config['outClasses'],1))
        TN = np.zeros((config['outClasses'],1))
        FP = np.zeros((config['outClasses'],1))
        FN = np.zeros((config['outClasses'],1))
        
        for data in testLoader:
            fileNos,labels = data
            labels = labels.tolist()
            images = torch.FloatTensor(config['batchSize'], config['inShape'][0],
                                       config['inShape'][1], config['inShape'][2])
            i = -1
            for file in fileNos:
                i += 1
                fileName = config['testDir']+str(file)+'.png'
                img = cv2.imread(fileName)
                if img is not None:
                    img = torch.from_numpy(img)
                    images[i] = img.permute(2,0,1)   # Channels, Height, Width
                else:
                    labels.pop(i)
                    fileNos.pop(i)
                    i -= 1
            images.to(device)
            
            probabilities = model(images)
            pred = np.array(torch.argmax(probabilities))
            
            i = -1
            for gt in labels:
                i += 1
                total[gt] += 1
                TN[gt] += 1
                if gt == pred[i]:
                    TP[gt] += 1
                    TN[gt] -= 1
                else:
                    FN[gt] += 1
                    FP[pred[i]] += 1
                    
        ACC = np.transpose(100*(TP + TN)/(total))
        PRE = np.transpose(100*(TP)/(TP + FP))
        REC = np.transpose(100*(TP)/(TP + FN))
        F1S = np.transpose(100*(2*REC*PRE)/(REC + PRE))
        
        acc = sum(TP+TN)/sum(total)
        pre = sum(TP*total)/sum((TP+FP)*total)
        rec = sum(TP*total)/sum((TP+FN)*total)
        f1s = sum(2*rec*pre)/sum(rec+pre)
        
        print('Accuracy: ' + str(acc) + '\nPrecision: ' + str(pre) +
              '\nRecall: ' + str(rec) + '\nF1Score: ' + str(f1s))
        OVR = np.array([acc, pre, rec, f1s])
        
        return np.array([ACC,PRE,REC,F1S,OVR])