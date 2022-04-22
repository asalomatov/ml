# train.py
import argparse 
import os
import joblib
import pandas as pd
from sklearn import metrics
import config
import model_dispatch
from ml_utils import binary_classification_metrics

def run(fold, model, targ_clm):
    # read the training data with folds 
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True) # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop(targ_clm, axis=1).values 
    y_train = df_train.label.values
    # similarly, for validation, we have
    x_valid = df_valid.drop(targ_clm, axis=1).values 
    y_valid = df_valid.label.values
    clf = model_dispatch.models[model]
    # fir the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy
    perf_metrics = binary_classification_metrics(y_valid, preds) 
    perf_metrics.insert(0, 'fold', fold)
    print(perf_metrics)
    # save the model
    joblib.dump( clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin") )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--fold", type=int )
    parser.add_argument( "--model", type=str )
    args = parser.parse_args()
    
    run( fold=args.fold, model=args.model )