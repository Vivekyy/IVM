import pandas as pd

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