import pandas as pd
import numpy as np
from sklearn import model_selection

def split_data(df, targ_clm, train_perc=0.7, rand_state=212):
    """Train/Validation split data frame.
    Return the split as dataframes."""
    # df = df.sample(frac=1).reset_index(drop=True)
    feature_clm = [i for i in df.columns if i != targ_clm]
    assert len(feature_clm) == len(df.columns) - 1, "Check feature/targ clms"
    X = df[feature_clm]
    y = df[targ_clm]
    X_trn, X_vld, y_trn, y_vld = model_selection.train_test_split(X, y,
                                                train_size=train_perc,
                                                random_state=rand_state,
                                                shuffle=True,
                                                stratify=y)
    X_trn.insert(loc=len(X_trn.columns),
                        column=y_trn.name,
                        value=y_trn)
    X_vld.insert(loc=len(X_vld.columns),
                        column=y_vld.name,
                        value=y_vld)

    return([X_trn.reset_index(drop=True), X_vld.reset_index(drop=True)])

def insert_folds(df, targ_clm, k_folds=5):
    """After abhishekkrthakur/approachingalmost"""
    df = df.sample(frac=1).reset_index(drop=True)
    df["kfold"] = -1
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=k_folds, )
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df[targ_clm])):
        df.loc[val_, 'kfold'] = fold
    return(df)

def create_folds_regr(df, targ_clm, k_folds=5):
    """After abhishekkrthakur/approachingalmost"""
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    # calculate the number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(df))))
    # bin targets
    df.loc[:, "bins"] = pd.cut(df[targ_clm], bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=k_folds)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, 'kfold'] = f
    df = df.drop("bins", axis=1)
    return(df)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=100)
    df =  pd.DataFrame(X, columns=['V%s' % str(i) for i in range(X.shape[1])])
    df.loc[:, 'targ'] = y
    l = split_data(df, 'targ')

    from sklearn.datasets import fetch_openml
    # diabetes dataset from openML
    diab = fetch_openml(data_id=37)
    df = diab.frame
    df.loc[:, 'class'] = df.loc[:, 'class'].map({'tested_positive': 1,
                                                'tested_negative': 0})
    l = split_data(df, 'class')

    df_trn = insert_fold(l[0], 'class')