# src/create_folds.py 

# Importing the libraries 
import pandas as pd
from sklearn.model_selection import StratifiedKFold 
import numpy as np 

# Creating the folds 
if __name__ == "__main__":
    df = pd.read_csv('../input/creditcard.csv')
    df['kfold'] = -1 
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.Class.values 

    kf = StratifiedKFold(n_splits=5)

    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,"kfold"] = f 

    # Saving the file 
    df.to_csv('../input/creditcard_fold.csv',index=False)
    print('Created Folds')