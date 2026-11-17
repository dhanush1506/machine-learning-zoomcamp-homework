import requests

url = 'http://localhost:9696/predict'

client = {
    "age": 35,
    "job": "blue-collar",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": -41,
    "housing": "yes",
    "loan": "no",
    "contact": "NA",
    "day_of_week": 2,
    "month": "jun",
    "campaign": 1,
    "pdays": 1,
    "previous": 1,
    "poutcome": "success"
 }

response = requests.post(url, json=client)
result = response.json()
print(result)