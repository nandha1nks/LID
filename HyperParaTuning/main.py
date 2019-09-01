'''
Main Wrapper to be called from the bash script
It takes in all arguements from the config.yaml file and runs
Train or Test or Predict with necessary preProcessing steps

Saving only weights after training
Testing saves the various metrics as np array

Need to write new code for prediction later on
'''
import argparse
import yaml
import CRNN
import train
import test
import torch
import numpy as np
import pickle
from createCombinations import createCombos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',dest="config", default="config.yaml")
    #config = load(open(parser.parse_args().config))
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fileName = createCombos()
    file = open('fileName','rb')
    combinations = pickle.load(file)
    file.close()
    
    for combo in combinations:
        print(combo.keys)
        for x,y in combo.items():
            config[x] = y
        config['runName'] = str(combinations.index(combo))
        
        model, lossHist = train.trainModel(config)
        torch.save(model.state_dict(), config['modelLoc'] + config['runName'] +
                   "Weights.pth")
        
        #Saves only weights (need to create model object and load this)
        #torch.save(model, config['modelLoc'] + config['runName'] + "Full.pth")
        #Saves entire Model (not advised, can break lot of ways due to directory issue)

        confMatrix, metrics = test.testModel(config)
        np.save(config['modelLoc'] + config['runName'] + "Confusion.npy",confMatrix)
        f = open(config['modelLoc'] + config['runName'] + "Metrics.txt",'w+')
        f.write(metrics)
        f.close()