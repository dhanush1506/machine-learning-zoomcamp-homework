import pickle
import pandas as pd

from flask import Flask
from flask import request, jsonify

app = Flask('brand-marketing')

model_file = 'model_C=0.1.bin'

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
    df.day_of_week.describe()
    df.rename(columns = {'day_of_week':'day_of_month'}, inplace=True)

    return df

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.route('/predict', methods = ['POST'])
def predict():
    client = request.get_json()

    df = prepare(pd.DataFrame([client]))
    X = dv.transform(df[categorical + numerical].to_dict(orient='records'))
    y_pred = model.predict_proba(X)[0,1]
    subscribe = y_pred >= 0.5

    result = {
        'subscription_probability:': float(y_pred),
        'subscribe:': bool(subscribe)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)