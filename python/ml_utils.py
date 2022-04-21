import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics

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

def create_folds(df, targ_clm, k_folds=5):
    """After abhishekkrthakur/approachingalmost"""
    df = df.sample(frac=1).reset_index(drop=True)
    df["kfold"] = -1
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=k_folds)
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df[targ_clm])):
        df.loc[val_, 'kfold'] = fold
    return(df)

def create_group_folds(df, targ_clm, group_clm, k_folds=5):
    """After abhishekkrthakur/approachingalmost"""
    df = df.sample(frac=1).reset_index(drop=True)
    df["kfold"] = -1
    # initiate the kfold class from model_selection module
    kf = model_selection.GroupKFold(n_splits=k_folds)
    for fold, (trn_, val_) in enumerate(kf.split(df,
                                            df[targ_clm],
                                            df[group_clm])):
        df.loc[val_, 'kfold'] = fold
    return(df)

def create_logo_folds(df, targ_clm, group_clm, k_folds=5):
    """After abhishekkrthakur/approachingalmost"""
    df = df.sample(frac=1).reset_index(drop=True)
    df["kfold"] = -1
    # initiate the kfold class from model_selection module
    kf = model_selection.LeaveOneGroupOut()
    for fold, (trn_, val_) in enumerate(kf.split(df,
                                            df[targ_clm],
                                            df[group_clm])):
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

def binary_classification_metrics(y_true, y_pred, y_prob=None):
    """ y_true - true labels
        y_pred - predicted labels
        y_prob - predicted probs"""
    confusion = metrics.confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    P = TP + FN
    N = TN + FP
    res = pd.DataFrame({
        'Prevalence': [P/(P + N)],
        'Sensitivity': [TP/P], # recall, TPR
        'Sensitivity_skl': [metrics.recall_score(y_true, y_pred)],
        'Specificity': [TN/N],
        'Accuracy': [(TP + TN)/(P + N)],
        'Accuracy_skl': [metrics.accuracy_score(y_true, y_pred)],
        'Precision': [TP/(TP + FP)], # also known as PPV
        'Precision_skl': [metrics.precision_score(y_true, y_pred)],
        'F1': [2*TP/(2*TP + FP + FN)],
        'F1_skl': [metrics.f1_score],
        'FPR': [FP/N], # 1 - specificity
        'MCC': [(TP * TN - FP * FN)/((TP + FP) * (FN + TN) * (FP + TN) * (TP + FN))**0.5],
        'MCC_skl': [metrics.metrics.matthews_corrcoef]
    })
    if y_prob is not None:
        res['AUC'] = metrics.roc_auc_score(y_true, y_prob)
        res['LogLoss'] = metrics.log_loss(y_true, y_prob)
    return(res)

def feature_clms(df_clms, targ_clm):
    """return list of feature columns"""
    res = [i for i in df_clms if i != targ_clm]
    return(res)


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
    df_trn, dr_val = split_data(df, 'class')
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=212).fit(
        df_trn[feature_clms(df_trn.columns, 'class')],
        df_trn['class'])
    y_pred = clf.predict(df_trn[feature_clms(df_trn.columns, 'class')])
    y_prob = clf.predict_proba(df_trn[feature_clms(df_trn.columns, 'class')])
    print(binary_classification_metrics(df_val['class'], y_pred))
    print(binary_classification_metrics(df_val['class'], y_pred, y_prob=y_prob))

    df_trn = create_folds(l[0], 'class')