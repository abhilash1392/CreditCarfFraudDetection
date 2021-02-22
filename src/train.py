# src/sc_logreg.py 
from inspect import TPFLAGS_IS_ABSTRACT
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import pickle
import argparse 
from model_dispatcher import models

def run(fold,model):
    df = pd.read_csv('../input/creditcard_fold.csv')

    features = [c for c in df.columns if c not in ('Class','kfold')]

    df_train = df[df.kfold !=fold].reset_index(drop=True)
    df_valid = df[df.kfold ==fold].reset_index(drop=True)

    X_train = df_train[features]
    X_valid = df_valid[features]
    y_train = df_train.Class.values 
    y_valid = df_valid.Class.values

    sc_columns = ['Time','Amount']

    preprocess = make_column_transformer((StandardScaler(),sc_columns),remainder='passthrough')

    
    

    clf = make_pipeline(preprocess,models[model])

    clf.fit(X_train,y_train)

    y_pred = clf.predict_proba(X_valid)[:,1]

    auc_score = roc_auc_score(y_valid,y_pred)

    print('Model----> {} | Fold ----> {} | AUC Score ----> {:.3f}'.format(model,fold,auc_score))

    filename = open(f'../models/{model}_{fold}.pkl','wb')
    pickle.dump(clf,filename)
    filename.close()
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int)
    parser.add_argument('--model',type=str)
    args = parser.parse_args()
    run(fold = args.fold,model = args.model)
    




        





