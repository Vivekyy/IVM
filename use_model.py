from model import Model
from utils import getDevice

import argparse
import pickle
import pandas as pd

import torch

device = getDevice()

def main(description, out_path, path, input_shape, dataset_path):
    key = pd.read_pickle('keys/' + path + '.pkl')
    output_shape = len(key.index)

    vec_path='vectorizers/' + path + '_vec.pkl'
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    model = Model(input_shape, output_shape)
    model.load_state_dict(torch.load('models/'+path+'.pt'))
    model.to(device)

    preds = getPreds(description, vectorizer, model)

    usable_preds = translatePreds(preds, key)

    out_descriptions = fetchDescriptions(usable_preds, dataset_path)

    output = pd.concat([usable_preds,out_descriptions], axis=1)
    output = output.apply(lambda x: pd.Series(x.dropna().values))

    output.to_excel(out_path)

    return output

def getPreds(description, vectorizer, model):

    with open (description, "r") as myfile:
        desc_str = myfile.read()

    X = vectorizer.transform([desc_str])
    X = torch.from_numpy(X.toarray())
    X = torch.squeeze(X).type(torch.FloatTensor)
    X = X.to(device)

    y_pred = model(X)

    return y_pred

def translatePreds(preds, key):

    preds = torch.round(preds)
    dfPreds = pd.DataFrame(preds.cpu().detach().numpy())
    dfPreds.columns = ['preds']

    dfCat = pd.concat([key,dfPreds], axis=1)

    idDF = dfCat.loc[dfCat['preds'] > 0.5]['Related Component ID'].to_frame()
    idDF.columns = ['Related Component ID']
    idDF = idDF.reset_index(drop=True)

    return idDF

def fetchDescriptions(idDF, dataset_path):
    
    
    dataDF = pd.read_excel(dataset_path)
    descDF = idDF.merge(dataDF, how='inner', on='Related Component ID').drop_duplicates(subset='Related Component ID')['Related Component Description']

    return descDF

def getArgs():
    parser = argparse.ArgumentParser(description='Get predicted dependent components')

    parser.add_argument('--description_path', help='The filename containing the description of the component you want to get predictions from (Default: description.txt)', default='description.txt', dest='description')
    parser.add_argument('--output_path', help='The filename you want to output the predictions to', default='Output.xlsx', dest='output_path')
    parser.add_argument('--model_name', help='The name of the model which you would like to use (Default: bow)', default='bow', dest='path')
    parser.add_argument('--vocab_size', help='The size of the vocabulary that the model was trained on (Default: 500)', default=500, dest='input_shape')
    parser.add_argument('--dataset_path', help='The path for the dataset you would like to access (Default: \'IntegratedValueModelrawdata.xlsx\')', default='IntegratedValueModelrawdata.xlsx', dest='dataset_path')

    args = parser.parse_args()

    return args.description, args.output_path, args.path, args.input_shape, args.dataset_path

if __name__ == "__main__":
    desc, out_path, path, input_shape, dataset_path = getArgs()
    output = main(desc, out_path, path, input_shape, dataset_path)

    print(output)