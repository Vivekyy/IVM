from vectorize import splitData, vectorize
from model import Model
from utils import CustomDataset, getDevice

import argparse
import numpy as np
import pandas as pd
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

def runEpoch(dataloader, model, loss_type, optimizer=None, desc=None):
    loss_counter = 0
    acc_counter = 0

    for _, batch in tqdm(enumerate(dataloader), desc = desc):
        X,y = batch['X'], batch['y']
        #print(y.size())
        X,y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_type(y_pred, y)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_counter += loss.item()
        acc_counter += getAcc(y_pred, y)

    mean_loss = loss_counter/len(dataloader)
    mean_acc = acc_counter/len(dataloader)

    return mean_loss, mean_acc

def getAcc(y_pred, y): #Necessary because we are going for multi-target accuracy
    preds = torch.round(y_pred)

    error = torch.sum(torch.abs(y - preds)).item()
    total = list(y.size())[0]*list(y.size())[1]

    acc = (total-error)/total

    return acc
    
def train(num_epochs, vec_type, input_shape, path, split=.8):
    start = time.perf_counter()

    X_train, X_test, y_train, y_test, y_map = splitData(split)
    y_map.to_pickle("keys/" + path + ".pkl")

    print("Vectorizing Data (%s)" % vec_type)
    tic = time.perf_counter()
    train_vecs, test_vecs = vectorize(X_train, X_test, vocab_size=input_shape, model_type = vec_type)
    toc = time.perf_counter()
    print(f"Done vectorizing data: {toc - tic:0.2f} seconds")
    print()

    trainData = CustomDataset(train_vecs, y_train)
    testData = CustomDataset(test_vecs, y_test)
    trainDL, valDL = trainValSplit(trainData, batch_size=10)

    output_shape = len(y_train.iloc[0])
    model = Model(input_shape, output_shape).to(device)

    loss_type = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

    best_accuracy=0
    for epoch in range(num_epochs):
        desc = str(vec_type) + ": Epoch " + str(epoch+1)
        
        model.train()
        train_loss, train_acc = runEpoch(trainDL, model, loss_type, optimizer=optimizer, desc = desc + " (Train)")

        print("Training Loss:", train_loss)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = runEpoch(valDL, model, loss_type, desc = desc + " (Validation)")
        lr_schedule.step(val_loss)

        print("Validation Loss:", val_loss)
        print("Validation Accuracy:", val_acc)

        if val_acc > best_accuracy:
            print("New Best Accuracy: Saving Epoch", epoch+1)
            best_accuracy = val_acc
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

    loss_type = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = runEpoch(testDL, model, loss_type, desc = "Testing Model")
    
    return test_loss, test_acc

def getArgs():
    parser = argparse.ArgumentParser(description='Train the prediction model')

    parser.add_argument('--path', help='What you would like the model to be named. For example, bow would be stored as models/bow.pt (Default: bow)', default='bow', dest='path')
    parser.add_argument('--epochs', help='The number of epochs you would like to train the model for (Default: 10)', default=10, dest='epochs')
    parser.add_argument('--vocab_size', help='The size of the vocabulary the model is using to make predictions (Default: 500)', default=500, dest='vocab_size')
    parser.add_argument('--split', help='The portion of data you would like to use as training data--the rest will be used as testing data. If you want to debug the model, input \'debug\' (Default: .8)', default=.8, dest='split')

    args = parser.parse_args()

    if args.split == 'debug': #Avoid overwriting existing models on debug run
        args.path = 'debug'

    return args.path, args.epochs, args.vocab_size, args.split

if __name__ == "__main__":
    path, epochs, vocab_size, split = getArgs()

    testData, traintime = train(epochs, 'BOW', vocab_size, path, split)
    test_loss, test_acc = test('models/' + path + '.pt', testData)
    print("Testing Loss: ", test_loss)
    print("Testing Acc: ", test_acc)
    print()

    print(f"Total Training Time: {traintime:0.2f}")
    print()

    '''
    key = pd.read_pickle('keys/' + path + '.pkl')
    print(key)
    print()
    '''