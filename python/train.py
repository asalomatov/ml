# train.py
import argparse 
import os
import joblib
import pandas as pd
from sklearn import metrics
import config
import model_dispatch
from ml_utils import binary_classification_metrics

def run(fold, model, targ_clm=config.TARG_CLM):
    # read the training data with folds 
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True) 
    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop([config.TARG_CLM, 'kfold'], axis=1).values 
    y_train = df_train[config.TARG_CLM].values
    # similarly, for validation, we have
    x_valid = df_valid.drop([config.TARG_CLM, 'kfold'], axis=1).values 
    y_valid = df_valid[config.TARG_CLM].values
    clf = model_dispatch.models[model]
    # fir the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    pred_prob = None
    try:
        pred_prob = clf.predict_proba(x_valid)[:, 1]
    except:
        # classifier cannot predict proba
        pass
    # calculate & print metrics
    if pred_prob is None:
        perf_metrics = binary_classification_metrics(y_valid, preds) 
    else:
        perf_metrics = binary_classification_metrics(y_valid, preds, pred_prob) 

    perf_metrics.insert(0, 'fold', fold)
    # print(perf_metrics)
    # save the model
    joblib.dump( clf,
        os.path.join(config.MODEL_OUTPUT, config.PREFIX + "_" + model + f"_{fold}.bin") )
    return(perf_metrics)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( "--fold", type=int )
    parser.add_argument( "--model", type=str )
    args = parser.parse_args()
    
    l = []
    for f in range(5):
        l.append(run( fold=f, model=args.model ))
    metr_df = pd.concat(l)
    print(metr_df)
    print(metr_df.AUC.mean())