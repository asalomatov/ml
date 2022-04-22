from sklearn.linear_model import LogisticRegression
from sklearn.svm import (SVC, NuSVC)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier)
import xgboost as xgb

RAND_STATE=212

models = {
    
    "logistic_regr_cls": LogisticRegression(),
    "svm_cls": SVC(probability=True, random_state=RAND_STATE),
    "nusvm_cls": NuSVC(probability=True, random_state=RAND_STATE),
    "decision_tree_cls_gini": DecisionTreeClassifier(criterion="gini", random_state=RAND_STATE ),
    "decision_tree_cls_entropy": DecisionTreeClassifier( criterion="entropy", random_state=RAND_STATE),
    "random_forest_cls": RandomForestClassifier(random_state=RAND_STATE),
    "gradient_boosting_cls": GradientBoostingClassifier(random_state=RAND_STATE),
    "extra_trees_cls": ExtraTreesClassifier(random_state=RAND_STATE),
    "xgboost_cls": xgb.XGBClassifier(
                                        objective="binary:logistic",
                                        eta=0.1,
                                        max_depth=3,
                                        gamma=1,
                                        alpha=1,
                                       #lambda=?,
                                        min_child_weight=1,
                                        subsample=1, #0.5,
                                        colsample_bytree=1, #0.7,
                                        colsample_bylevel=1, #0.7,
                                        n_estimators=200,
                                        random_state=RAND_STATE, 
                                        seed=RAND_STATE,
                                        n_jobs=-1,
                                        use_label_encoder=False,
                                        eval_metric="auc"
                                    ),


}

if __name__ == "__main__":
    pass