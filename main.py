# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:06:06 2019

@author: nandha
"""
import argparse
import yaml

from train import trainModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',dest="config", default="config.yaml")
    #config = load(open(parser.parse_args().config))
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    print(config)