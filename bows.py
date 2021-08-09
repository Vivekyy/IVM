from vectorize import splitData, vectorize
from model import Model
from utils import CustomDataset

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def trainValSplit(dataset, batchSize):
    randIndeces = np.random.permutation(len(dataset))
    trainIndeces = randIndeces[:int(.9*len(dataset))]
    valIndeces = randIndeces[int(.9*len(dataset)):]

    trainLoader = DataLoader(dataset, batch_size=batchSize, drop_last=True, 
                            sampler=SubsetRandomSampler(trainIndeces),
                            num_workers = 1, pin_memory=True)
    valLoader = DataLoader(dataset, batch_size=batchSize, drop_last=False, 
                            sampler=SubsetRandomSampler(valIndeces),
                            num_workers = 1, pin_memory=True)

    return trainLoader, valLoader

def runEpoch(dataloader, model, loss_type, optimizer=None, desc=None):
    loss_counter = 0
    acc_counter = 0

    for X, y in tqdm(dataloader, desc = desc):
        X,y = X.to(device), y.to(device)

        y_pred = model(y)

        loss = loss_type(y_pred, y)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_counter += loss.item()
        acc_counter += (y_pred.max(1)[1] == y).float().mean().item()

    mean_loss = loss_counter/len(dataloader)
    mean_acc = acc_counter/len(dataloader)

    return mean_loss, mean_acc
    
def train(num_epochs, vec_type, input_shape, path):
    X_train, X_test, y_train, y_test = splitData()
    train_vecs, test_vecs = vectorize(X_train, X_test, vocab_size=input_shape, model_type = vec_type)

    trainData = CustomDataset(train_vecs, y_train)
    testData = CustomDataset(test_vecs, y_test)
    trainDL, valDL = trainValSplit(trainData, batch_size=10)

    output_shape = len(y_train.iloc[0])
    print(y_train)
    print(output_shape)
    model = Model(input_shape, output_shape).to(device)

    loss_type = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

    best_accuracy=0
    for epoch in range(num_epochs):
        desc = vec_type + ": Epoch " + epoch
        
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
            print("New Best Accuracy: Saving Epoch", epoch)
            best_accuracy = val_acc
            torch.save(model.state_dict(), "models/" + path)
    
    return testData

def test(model_path, testData):
    X,y = testData
    testDL = DataLoader(testData, batch_size=10, num_workers=1, pin_memory=True)

    input_shape = len(X.iloc[0])
    output_shape = len(y.iloc[0])
    model = Model(input_shape, output_shape)
    model.load_state_dict(torch.load(model_path))

    loss_type = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = runEpoch(testDL, model, loss_type, desc = "Testing Model")
    
    return test_loss, test_acc

if __name__ == "__main__":
    train(10, 'tfidf', 20, 'tfidf.pt')
    test('models/tfidf')