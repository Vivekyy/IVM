#Cudf can replace pandas for potential speedup, not using cudf due to compatibility issues with the environment setup
#import cudf

import pandas as pd
import scipy

import torch
from torch.utils.data import Dataset

def getDevice():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def getData(dataset_path):
    df = pd.read_excel(dataset_path)
    
    df = df[['Component', 'Component ID', 'Related Component', 'Related Component ID', 'Component Description', 'Related Component Description']]

    return df

def getXy(split, dataset_path):
    df = getData(dataset_path)

    df.sample(frac=1).reset_index(drop=True) #Shuffles data and resets indices

    X = df.groupby('Component ID')['Component Description'].max()

    if split=='debug': #for debugging
        split=.05
    
    y_dummies = pd.get_dummies(df['Related Component ID']) #.iloc[:int(split*len(X))] 
    #add above as attempt at doing only train-relevant related components
    #omitted because the model seems to perform well regardless

    y_map = y_dummies.columns.to_frame(index=False, name='Related Component ID')

    df2 = pd.concat([df['Component ID'],y_dummies], axis=1)

    grouped_df = df2.groupby('Component ID').sum()

    return X, grouped_df, y_map

class CustomDataset(Dataset):

    def __init__(self, X, y, transform=None):
        
        self.X = X
        self.y = y
        
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Handle different input datatypes and convert to tensors
        if isinstance(self.X, pd.DataFrame):
            X_idx = torch.from_numpy(self.X.iloc[idx].to_numpy())
        else:
            X_idx = self.X[idx]
        
        if isinstance(X_idx, scipy.sparse.csr.csr_matrix):
            X_idx = torch.from_numpy(X_idx.toarray())
        
        if isinstance(self.y, pd.DataFrame):
            y_idx = torch.from_numpy(self.y.iloc[idx].to_numpy())
        else:
            y_idx = self.y[idx]
        
        if isinstance(y_idx, scipy.sparse.csr.csr_matrix):
            y_idx = torch.from_numpy(y_idx.toarray())
        
        X_idx, y_idx = torch.squeeze(X_idx).type(torch.FloatTensor), y_idx.type(torch.FloatTensor)
        
        #Apply transformations
        if self.transform:
            X_idx = self.transform(X_idx)
            y_idx = self.transform(y_idx)

        sample = {'X': X_idx, 'y': y_idx}

        return sample