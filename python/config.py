
MODEL_OUTPUT = "../models/"

# PREFIX = "mnist"
# TRAINING_FILE = "../input/mnist_train_folds.csv" 
# VALIDATION_FILE = "../input/mnist_validate.csv" 

# PREFIX = "diabetes"
# INPUT_DATA = "../input/diabetes.csv"
# TARG_CLM = "Outcome"
# TRAINING_FILE = "../input/diabetes_train_folds.csv" 
# VALIDATION_FILE = "../input/diabetes_validate.csv" 

# PREFIX = "sbPCR"
# INPUT_DATA = "../input/ALL_IA_DL_features.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "sbPCR_2010Nature.features"
# INPUT_DATA = "../input/2010_nature.features.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "HU_IA_DLfeatures"
# INPUT_DATA = "../input/HU_IA_DLfeatures.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "HU_IA.features"
# INPUT_DATA = "../input/HUM_IA.features.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "sbPCR_2010Nature_protfeat"
# INPUT_DATA = "../input/protfeat_2010.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "sbPCR_2010Nature_DLfeatures"
# INPUT_DATA = "../input/2010Nature_DLfeatures.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "sbPCR_2010Nature_protfeat_nna"
# INPUT_DATA = "../input/protfeat_nna_2010.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

# PREFIX = "sbPCR_2010Nature_DL_protfeat_nna"
# INPUT_DATA = "../input/DL_protfeat_nna_2010.csv"
# TARG_CLM = "lbl"
# TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
# VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

PREFIX = "sbPCR_2010Nature_DLres20_protfeat_nna"
INPUT_DATA = "../input/DLres20_protfeat_nna_2010.csv"
TARG_CLM = "lbl"
TRAINING_FILE = "../input/%s_train_folds.csv" % PREFIX 
VALIDATION_FILE = "../input/%s_validate.csv" % PREFIX 

FEATURE_SET_PROT = ['helix', 'loop', 'sheet', 'sasa_min_norm', 'sasa_max_norm',
       'regmotifs', 'psp', 'ligbindsites', 'ptm', 'disulfide',
       'max_bfactor_calc', 'min_bfactor_calc', 'bfactor_calc_max_min_ratio',
       'effectiveness_norm', 'sensitivity_norm', 'n_acidic_res',
       'n_amphipathic_res', 'n_aromatic_res', 'n_basic_res', 'n_disulfide',
       'n_hbond_a', 'n_hbond_d', 'n_hydrophobic_res', 'n_polar_res',
       'n_water']
FEATURE_SET_DL = ['V' + str(i) for i in range(1280)]
FEATURE_SET = FEATURE_SET_DL
# FEATURE_SET = FEATURE_SET_PROT
# FEATURE_SET = FEATURE_SET_PROT + FEATURE_SET_DL
PLT_TITLE = "DLres20 features"

PCA_COMPONENTS = 2000
PCA_KERNEL = 'linear' # 'linear', ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’