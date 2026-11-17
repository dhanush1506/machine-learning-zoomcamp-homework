import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

C=0.1
output_file = f'model_C={C}.bin'

categorical = ['job', 'marital', 'education', 'default', 'housing', 
               'loan', 'contact', 'month','poutcome', 'contacted_before']
numerical = ['age', 'balance', 'day_of_month', 'campaign', 'pdays', 'previous']

def prepare(df):

    df = df.copy()
    strings = list(df.dtypes[df.dtypes == 'str'].index)
    for s in strings:
        df[s] = df[s].str.lower().str.replace(' ', '_')

    nulls = list(df.columns[df.isnull().any()])
    for n in nulls:
        df[n] = df[n].fillna('NA')

    df['contacted_before'] = (df['pdays']!=-1).astype(int)
    df['pdays'] = df['pdays'].replace(-1, 999)
    df.rename(columns = {'day_of_week':'day_of_month'}, inplace=True)

    return df

def train(df_train, y_train, C):

    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train,y_train)

    return dv, model


def predict(df, dv, model):

    dicts = df[categorical+numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred

#Loading data
print('Loading data...')

bank_marketing = fetch_ucirepo(id=222)
df = prepare(bank_marketing.data.features)
df['convert'] = (bank_marketing.data.targets['y'] == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)

#training the model
print(f'Training the model with C={C}')

dv, model = train(df_full_train, df_full_train.convert.values, C=C)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(df_test.convert.values, y_pred)
print(f'auc={auc}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')