import pandas as pd

import torch
from torch.utils.data import Dataset

def getData():
    df = pd.read_excel('IntegratedValueModelrawdata.xlsx')
    
    df = df[['Component', 'Component ID', 'Related Component', 'Related Component ID', 'Component Description', 'Related Component Description']]

    return df

def getXy():
    df = getData()

    X = df.groupby('Component ID')['Component Description'].max()

    y_dummies = pd.get_dummies(df['Related Component ID'])

    df2 = pd.concat([df['Component ID'],y_dummies], axis=1)

    grouped_df = df2.groupby('Component ID').sum()

    return X, grouped_df

class CustomDataset(Dataset):

    def __init__(X, y, self, transform=None):
        
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'X': self.X.iloc[idx], 'y': self.y.iloc[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample