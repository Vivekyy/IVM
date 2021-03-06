from vectorize import splitData, vectorize
from model import Model
from utils import CustomDataset, getDevice

import argparse
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

device = getDevice()

def trainValSplit(dataset, batch_size):
    randIndeces = np.random.permutation(len(dataset))
    trainIndeces = randIndeces[:int(.9*len(dataset))]
    valIndeces = randIndeces[int(.9*len(dataset)):]

    trainLoader = DataLoader(dataset, batch_size=batch_size, drop_last=True, 
                            sampler=SubsetRandomSampler(trainIndeces),
                            num_workers = 1, pin_memory=True)
    valLoader = DataLoader(dataset, batch_size=batch_size, drop_last=False, 
                            sampler=SubsetRandomSampler(valIndeces),
                            num_workers = 1, pin_memory=True)

    return trainLoader, valLoader

def runEpoch(dataloader, model, loss_type, weight=1, optimizer=None, desc=None):
    loss_counter = 0
    acc_counter = 0
    sens_counter = 0

    for _, batch in tqdm(enumerate(dataloader), desc = desc):
        X,y = batch['X'], batch['y']
        #print(y.size())
        X,y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_type(y_pred, torch.mul(y,weight)) #Experimenting with weighting to decrease false negatives

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_counter += loss.item()
        acc_counter += getAcc(y_pred, y)
        sens_counter += getSensitivity(y_pred, y)

    mean_loss = loss_counter/len(dataloader)
    mean_acc = acc_counter/len(dataloader)
    mean_sens = sens_counter/len(dataloader)

    return mean_loss, mean_acc, mean_sens

def getAcc(y_pred, y): #Necessary because we are going for multi-target accuracy
    preds = torch.where(y_pred>0.5, 1, 0).type(torch.FloatTensor).to(device)

    error = torch.sum(torch.abs(y - preds)).item()
    total = torch.numel(y)

    acc = (total-error)/total

    return acc

def getSensitivity(y_pred, y):
    preds = torch.where(y_pred>0.5, 1, 0).type(torch.FloatTensor).to(device)

    hits=0
    for i in range(len(y)):
        hits += torch.dot(preds[i],y[i]).item()

    all_pos = torch.sum(y).item()

    sens = hits/all_pos

    return sens

    
def train(num_epochs, vec_type, input_shape, path, weight, split=.8, dataset_path='IntegratedValueModelrawdata.xlsx'):
    start = time.perf_counter()

    X_train, X_test, y_train, y_test, y_map = splitData(split, dataset_path)
    y_map.to_pickle("keys/" + path + ".pkl")

    print("Vectorizing Data (%s)" % vec_type)
    tic = time.perf_counter()
    vec_path = "vectorizers/"+path+"_vec.pkl"
    train_vecs, test_vecs = vectorize(X_train, X_test, vec_path, vocab_size=input_shape, model_type = vec_type)
    toc = time.perf_counter()
    print(f"Done vectorizing data: {toc - tic:0.2f} seconds")
    print()

    trainData = CustomDataset(train_vecs, y_train)
    testData = CustomDataset(test_vecs, y_test)
    trainDL, valDL = trainValSplit(trainData, batch_size=10)

    output_shape = len(y_train.iloc[0])
    model = Model(input_shape, output_shape).to(device)

    #vector with length = output_shape = number of classes filled with value = weight
    weighting = torch.mul(torch.ones(output_shape), weight).to(device)

    loss_type = nn.BCEWithLogitsLoss(pos_weight=weighting)
    optimizer = optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

    best_sens=0
    for epoch in range(num_epochs):
        desc = str(vec_type) + ": Epoch " + str(epoch+1)
        
        model.train()
        train_loss, train_acc, train_sens = runEpoch(trainDL, model, loss_type, optimizer=optimizer, desc = desc + " (Train)")

        print("Training Loss:", train_loss)
        print("Training Sensitivity:", train_sens)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_sens = runEpoch(valDL, model, loss_type, desc = desc + " (Validation)")
        lr_schedule.step(val_loss)

        print("Validation Loss:", val_loss)
        print("Validation Accuracy:", val_acc)
        print("Validation Sensitivity:", val_sens)

        if val_sens > best_sens:
            print("New Best Sensitivity: Saving Epoch", epoch+1)
            best_sens = val_sens
            torch.save(model.state_dict(), "models/" + path + ".pt")
        
        print()
    
    stop = time.perf_counter()
    traintime = stop - start
    return testData, traintime

def test(model_path, testData):
    testDL = DataLoader(testData, batch_size=10, num_workers=1, pin_memory=True)

    dataloader_iterator = iter(testDL)
    batch = next(dataloader_iterator)
    X,y = batch['X'], batch['y']

    input_shape = list(X.size())[1]
    output_shape = list(y.size())[1]

    model = Model(input_shape, output_shape)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    loss_type = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, test_sens = runEpoch(testDL, model, loss_type, desc = "Testing Model")
    
    return test_loss, test_acc, test_sens

def getArgs():
    parser = argparse.ArgumentParser(description='Train the prediction model')

    parser.add_argument('--path', help='What you would like the model to be named. For example, bow would be stored as models/bow.pt (Default: bow)', default='bow', dest='path')
    parser.add_argument('--epochs', help='The number of epochs you would like to train the model for (Default: 10)', default=10, dest='epochs', type=int)
    parser.add_argument('--vocab_size', help='The size of the vocabulary the model is using to make predictions (Default: 500)', default=500, dest='vocab_size', type=int)
    parser.add_argument('--tfidf', help='Include this tag if you would like to train a TF-IDF model as opposed to the standard BOW model', action='store_const', const='TFIDF', default='BOW', dest='vec_type')
    parser.add_argument('--weight', help='A weighting of the loss that corresponds to weighting false negatives. A higher weight will result in more recommendations (Default: 20)', default=20, dest='weight', type=int)
    parser.add_argument('--split', help='The portion of data you would like to use as training data--the rest will be used as testing data. (Default: .8)', default=.8, dest='split', type=float)
    parser.add_argument('--dataset_path', help='The path for the dataset you would like to access (Default: \'IntegratedValueModelrawdata.xlsx\')', default='IntegratedValueModelrawdata.xlsx', dest='dataset_path')
    parser.add_argument('--debug', help='Include this tag if you would like to do a faster trial run of the code (the ouput model will not be useful).', action='store_const', const=True, default=False, dest='debug')

    args = parser.parse_args()
    if args.vec_type == 'TFIDF' and args.path == 'bow':
        args.path = 'tfidf'

    if args.debug == True:
        args.split = 'debug'
        args.path = 'debug' #Avoid overwriting existing models on debug run

    return args.path, args.epochs, args.vocab_size, args.vec_type, args.weight, args.split, args.dataset_path

if __name__ == "__main__":
    path, epochs, vocab_size, vec_type, weight, split, dataset_path = getArgs()

    testData, traintime = train(epochs, vec_type, vocab_size, path, weight, split, dataset_path)
    test_loss, test_acc, test_sens = test('models/' + path + '.pt', testData)
    print("Testing Loss: ", test_loss)
    print("Testing Accuracy: ", test_acc)
    print("Testing Sensitivity: ", test_sens)
    print()

    print(f"Total Training Time: {traintime:0.2f}")
    print()

    #For printing out the saved key of Dependent Component IDs
    '''
    key = pd.read_pickle('keys/' + path + '.pkl')
    print(key)
    print()
    '''