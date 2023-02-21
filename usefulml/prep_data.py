import config
import general_utils
import ml_utils
import pandas as pd
import os

# read
df = pd.read_csv(config.INPUT_DATA)

# transform
# to-do

# split
df_trn, df_val = ml_utils.split_data(df, config.TARG_CLM, train_perc=0.9)
df_trn = ml_utils.create_folds(df_trn, config.TARG_CLM, k_folds=10)

# mkdir -p
general_utils.run_in_shell("mkdir -p " + config.MODEL_OUTPUT)
general_utils.run_in_shell("mkdir -p " + os.path.dirname(config.TRAINING_FILE))

# save
df_trn.to_csv(config.TRAINING_FILE, index=False)
df_val.to_csv(config.VALIDATION_FILE, index=False)