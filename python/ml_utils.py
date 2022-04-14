import pandas as pd
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

    def save_asdf():
        """Save output split_data to dataframes"""
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