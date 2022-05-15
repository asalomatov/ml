# train.py
import argparse 
import os
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
import config
import model_dispatch
from ml_utils import binary_classification_metrics
from ml_utils import roc_curve
from matplotlib import pyplot as plt


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
        roc_crv =roc_curve(y_valid, pred_prob)

    perf_metrics.insert(0, 'fold', fold)
    # print(perf_metrics)
    # save the model
    joblib.dump( clf,
        os.path.join(config.MODEL_OUTPUT, config.PREFIX + "_" + model + f"_{fold}.bin" ) )
    return([perf_metrics, roc_crv])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument( "--fold", type=int )
    parser.add_argument( "--model", type=str )
    args = parser.parse_args()
    
    plt.figure(figsize=(10,10))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    l = []
    l_roc = []
    for f in range(5):
        metr, roc_crv = run( fold=f, model=args.model )
        tprs.append(np.interp(mean_fpr, roc_crv.fpr, roc_crv.tpr))
        tprs[-1][0] = 0.0
        plt.plot(roc_crv.fpr, roc_crv.tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (f, metr.AUC.iloc[0]))
        l.append(metr)
        l_roc.append(roc_crv)

    metr_df = pd.concat(l)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(metr_df.AUC)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('CYS reactivity prediction\nCross-Validation ROC, DL features, 2010 Nature data', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.savefig(os.path.join(config.MODEL_OUTPUT, args.model + '_roc.png'), bbox_inches = "tight")
    plt.show()
    print(metr_df)
    print(metr_df.AUC.mean())