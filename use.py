from model import Model
from utils import getDevice

import argparse
import pandas as pd

import torch

device = getDevice()

def main(description, path, input_shape):
    key = pd.read_pickle('keys/' + path + '.pkl')
    output_shape = len(key.index)

    model = Model(input_shape, output_shape)
    model.load_state_dict(torch.load('models/'+path+'.pt'))
    model.to(device)

    preds = getPreds(description, model)

    usable_preds = translatePreds(preds, key)

    descriptions = fetchDescriptions(usable_preds)

    output = pd.concat([usable_preds,descriptions], axis=1)

    return output

def getPreds(description, model):

    return y_pred

def translatePreds(preds, key):

    return idDF

def fetchDescriptions(idDF):

    return descriptionDF

def getArgs():
    parser = argparse.ArgumentParser(description='Get predicted dependent components')

    parser.add_argument('description', help='The description of the component you want to get predictions from')
    parser.add_argument('--model_path', help='The model which you would like to use (Default: bow)', default='bow', dest='path')
    parser.add_argument('--input_shape', help='The size of the vocabulary that the model was trained on (Default: 500)', default=500, dest='input_shape')

    args = parser.parse_args()

    return args.description, args.path, args.input_shape

if __name__ == "__main__":
    desc, path, input_shape = getArgs()
    output = main(desc, path, input_shape)

    print(output)