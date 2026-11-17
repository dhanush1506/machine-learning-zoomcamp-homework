# Bank Marketing - Term Deposit Prediction

## Problem description

A Portuguese bank runs phone call campaigns to get clients to sign up for a term deposit. Most clients say no. Only 11.7% of the clients in the data signed up. So a lot of calls are made to people who are not interested.

The goal of this project is to predict whether a client will sign up for a term deposit before calling them. With this, the bank can call the clients who are most likely to sign up first, and save time and effort on the others.

The model gives the probability that a client will sign up. It is served as a web service, so the bank can send a client's details and get a prediction back.

## Dataset

The data is the Bank Marketing dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/222/bank+marketing

It has 45,211 clients and 16 features, like age, job, balance, how the client was contacted, and the result of the previous campaign. The target is `y`, whether the client signed up (yes/no).

The data is loaded directly in the code using the `ucimlrepo` package, so there is no need to download it separately. An internet connection is needed when running the notebook or `train.py`.

**Why `duration` was dropped:** `duration` is the length of the last call. It has a very high AUC (0.81) because interested clients talk longer. But the duration is known only after the call ends, when the outcome is already known. So it can't be used to decide whom to call. The dataset description also says it should be dropped for a realistic model. So it is not used in the model.

## Approach

- Filled missing values in the categorical columns with 'NA'.
- `pdays` uses -1 for clients who were not contacted before. Since -1 can be read as a recent contact, it was replaced with 999 and a new column `contacted_before` was added.
- Renamed `day_of_week` to `day_of_month`, since the values are 1-31.
- Checked feature importance with mutual information (categorical) and ROC AUC (numerical).
- Trained a logistic regression model and checked the effect of removing each feature. Removing features did not improve the model, so all features were kept.
- Tuned `C` using 5-fold cross validation. C=0.1 gave the best mean AUC.
- The final model gives an AUC of 0.774 on the test set.

## Files

- `Bank-marketing.ipynb` - data preparation, EDA, model training and tuning
- `train.py` - trains the final model and saves it to `model_C=0.1.bin`
- `model_C=0.1.bin` - the saved model
- `predict.py` - web service that loads the model and gives predictions
- `predict-test.py` - sends a sample client to the service and prints the result
- `Pipfile`, `Pipfile.lock` - dependencies
- `Dockerfile` - to run the service in a container

## How to run

### 1. Install the dependencies

The project uses pipenv. Install pipenv if you don't have it:

```bash
pip install pipenv
```

Then install the dependencies from the project folder:

```bash
pipenv install
```

To activate the environment:

```bash
pipenv shell
```

### 2. Open the notebook (optional)

The notebook needs a few extra packages (Jupyter, matplotlib, seaborn). Install them with:

```bash
pipenv install --dev
```

Then open the notebook:

```bash
pipenv run jupyter notebook
```

### 3. Train the model

```bash
python train.py
```

This trains the model, prints the test AUC and saves the model to `model_C=0.1.bin`.

### 4. Run the service locally

```bash
python predict.py
```

In another terminal, run the test:

```bash
python predict-test.py
```

## Run with Docker

Build the image:

```bash
docker build -t bank-marketing .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 bank-marketing
```

Then in another terminal run the test:

```bash
python predict-test.py
```

## Testing the service

The service takes a client's details as JSON and returns the probability of signing up and a yes/no decision.

Output of `predict-test.py` while the Docker container is running:

<img width="875" height="203" alt="Screenshot 2026-11-17 104619" src="https://github.com/user-attachments/assets/405385c4-bcd7-46c8-8653-1c20251829e0" />

## Cloud deployment

The service is deployed on Render using the Dockerfile in this folder: https://<your-url>.onrender.com

Steps used:
1. On Render: New → Web Service → connected this GitHub repo
2. Root Directory: `Mid-Term-Project`, Language: Docker, Instance: Free
3. Environment variable: `PORT=9696`

The service running on Render:



Testing the deployed service:

```bash
PREDICT_URL=https://machine-learning-zoomcamp-homework-1xrg.onrender.com/predict pipenv run python predict-test.py
```

