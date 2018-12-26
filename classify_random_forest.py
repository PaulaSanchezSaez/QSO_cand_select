#code to predict light curve classes using random Forest

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import pickle
import sys, getopt
import ast
from scipy.stats import randint as sp_randint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

###############################################################################
#modify these and only these variables:

ncores = 2 #number of cores used to compute the features

training_file = 'var_features_Dec2018/features_train_samp_Oct2018_well_sampled.txt' # file with var features of the labeled objects

test_file = 'var_features_Dec2018/features_COSMOS_Dec2018_well_sampled.txt' # file with var features of the unlabeled objects

output = 'classification_Dec2018/classification_noDRW_COSMOS.csv' #Output text file with predicted classes

dump_model = "False" # If True dump the rf model into a file

model_file = 'rf_model_nocolor_noDRW_accuracy_Dec18'#'rf_model_nocolor_Dec18' # where the rf model is written

stats_file = 'rf_stat_nocolor_noDRW_accuracy_Dec18'#'rf_stat_nocolor_Dec18' # where the statistics of the model are saved

gen_conf_matrix = "False" #If True generate the confusion matrix

conf_matrix_name = 'confusion_matrix_rf_nocolor_noDRW_Dec2018.pdf'

###############################################################################
# To modify the parameters from the terminal
myopts, args = getopt.getopt(sys.argv[1:],"n:t:p:o:d:m:s:g:c:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-n':
        ncores = a
    elif o == '-t':
        training_file = a
    elif o == '-p':
        test_file = a
    elif o == '-o':
        output = a
    elif o == '-d':
        dump_model = a
    elif o == '-m':
        model_file = a
    elif o == '-s':
        stats_file = a
    elif o == '-g':
        gen_conf_matrix = a
    elif o == '-c':
        conf_matrix_name = a



ncores = int(ncores)

training_file = str(training_file)

test_file = str(test_file)

output = str(output)

dump_model = ast.literal_eval(dump_model)

model_file = str(model_file)

stats_file = str(stats_file)

gen_conf_matrix = ast.literal_eval(gen_conf_matrix)

conf_matrix_name = str(conf_matrix_name)

###############################################################################
#features to consider in the classification.

feature_list = [
    #'zspec',
    #'TYPE',
    #'num_epochs',
    #'time_range',
    #'time_rest',
    #'umag',
    #'gmag',
    #'rmag',
    #'imag',
    #'zmag',
    #'u_g',
    #'g_r',
    #'r_i',
    #'i_z',
    #'tau',
    #'tau_lo',
    #'tau_up',
    #'sigma',
    #'sigma_lo',
    #'sigma_up',
    'P_var',
    'exvar',
    #'exvar_err',
    'A_mcmc',
    #'A_low',
    #'A_up',
    'gamma_mcmc',
    #'gamma_low',
    #'gamma_up',
    'best_period',
    #'peak',
    #'sig5',
    #'sig1',
    #'Mean',
    'Std',
    'Meanvariance',
    'MedianBRP',
    #'Rcs',
    #'PeriodLS',
    #'Period_fit',
    #'Color',
    'Autocor_length',
    #'SlottedA_length',
    'StetsonK',
    #'StetsonK_AC',
    'Eta_e',
    #'Amplitude',
    'PercentAmplitude',
    'Con',
    'LinearTrend',
    'Beyond1Std',
    #'FluxPercentileRatioMid20',
    #'FluxPercentileRatioMid35',
    #'FluxPercentileRatioMid50',
    #'FluxPercentileRatioMid65',
    #'FluxPercentileRatioMid80',
    #'PercentDifferenceFluxPercentile',
    'Q31',
    ]


###############################################################################
#loading the training DataFrame

print "Reading training data in \"%s\"" % training_file
training_data = pd.read_csv(training_file)

print "Pre-processing training data"
training_X = training_data[feature_list].astype('float64')
features = training_X.columns.values.tolist()

def tag_qso(label):
    if label != 'QSO':
        return 'NON-QSO'
    return label

label_encoder = preprocessing.LabelEncoder()
training_Y = training_data['TYPE'].apply(tag_qso)
label_encoder.fit(training_Y)
training_Y = label_encoder.transform(training_Y)

try:
    scaler = preprocessing.StandardScaler().fit(training_X)
except ValueError:
    column_is_invalid = training_X.applymap(lambda x: x==np.inf).any()
    invalid_columns = column_is_invalid[column_is_invalid].index.tolist()
    raise ValueError("Column(s) %s has(have) invalid values. Please exclude from feature list or remove respective rows." % invalid_columns)
training_X = scaler.transform(training_X)

#spliting the labeled set into a trainig set and a validation set
print "splitting training data"
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(training_X, training_Y, test_size=0.3)


###############################################################################
#do if dump_model is True

if dump_model:


    ###############################################################################
    #selection of RF hyper-parameters by cross validation
    print "Selecting hyper-parameters"

    param_dist = {"n_estimators": sp_randint(100, 500), "max_features": ['auto', 'sqrt']}
    model = ensemble.RandomForestClassifier(class_weight='balanced_subsample', n_jobs=ncores)

    n_iter_search = 100
    rf_model = model_selection.RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter_search, scoring='accuracy',cv=5)#probar accuracy y precision

    rf_model.fit(X_train, y_train)

    print "Model selected: \"%s\"" % rf_model.best_estimator_

    print "Best score: \"%s\"" % rf_model.best_score_

    print "Best param: \"%s\"" % rf_model.best_params_

    ###############################################################################
    #testing model performance
    print "testing model performance"

    y_true, y_pred = y_valid, rf_model.predict(X_valid)

    try:
        y_score = rf_model.decision_function(X_valid)
    except AttributeError:
        y_score = rf_model.predict_proba(X_valid)[:,1]

    f1_score = metrics.f1_score(y_true, y_pred)
    accuracy_score = metrics.accuracy_score(y_true, y_pred)
    precision_score = metrics.precision_score(y_true, y_pred)
    recall_score = metrics.recall_score(y_true, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_true, y_score)

    scores = { 'f1': f1_score, 'accuracy': accuracy_score, 'precision_score': precision_score, 'recall_score': recall_score, 'roc_auc_score': roc_auc_score}

    print "model performance: "
    print "f1 = %f " % (f1_score)
    print "accuracy = %f " % (accuracy_score)
    print "precision = %f " % (precision_score)
    print "recall = %f " % (recall_score)
    print "roc_auc = %f " % (roc_auc_score)

    ###############################################################################
    #generating confusion matrix

    if gen_conf_matrix:

        cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)

        plt.figure()
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

        plt.colorbar()
        tick_marks = np.arange(len(label_encoder.classes_))
        plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, label_encoder.classes_)

        print "Confusion matrix"
        print cnf_matrix

        fmt = '.2f'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel(u'True Label')
        plt.xlabel(u'Predicted Label')


        plt.savefig(conf_matrix_name, bbox_inches='tight')
        print "Confusion matrix saved in file \"%s\"" % conf_matrix_name

    ###############################################################################
    #saving model

    print "Dumping trained model in file \"%s\"" % model_file
    with open(model_file, 'wb') as pickle_file:
        model_dump = {
            'rf_model': rf_model,
            'scaler': scaler,
            'scores': scores,
            'label_encoder': label_encoder,
            'features': features,
            }
        pickle.dump(model_dump, pickle_file)

###############################################################################
#do if dump_model is False
else:

    if gen_conf_matrix:
        print "Cannot compute confusion matrix without training a model."

    print "Loading trained model in file \"%s\"." % model_file
    with open(model_file, 'rb') as pickle_file:
        model_dump = pickle.load(pickle_file)
        rf_model = model_dump['rf_model']
        scaler = model_dump['scaler']
        scores = model_dump['scores']
        label_encoder = model_dump['label_encoder']
        features = model_dump['features']

###############################################################################
#gen statistincs of the model_dump

with open(stats_file, 'wb') as statsfile:
    for key, value in scores.items():
        statsfile.write("Classification score \"%s\": %f\n" % (key, value))
    statsfile.write("\n")
    statsfile.write("Ranking of feature relevance for classification:\n")
    feature_relevance = list()

    feat_rel_val = rf_model.best_estimator_.feature_importances_

    for feature, score in zip(features, feat_rel_val):
        feature_relevance.append((feature, abs(score)))
    feature_relevance.sort(key=lambda x: x[1], reverse=True)
    for feature, score in feature_relevance:
        statsfile.write("%s: %s\n" % (feature, score))

###############################################################################
#loading the test DataFrame

print "Reading test data in \"%s\"" % test_file
test_data = pd.read_csv(test_file)

print "Pre-processing test data"
test_X = test_data[feature_list].astype('float64')
test_X = scaler.transform(test_X)

###############################################################################
#predicting classes of unlabel data

print "Classifying test data"
test_Y = label_encoder.inverse_transform(rf_model.predict(test_X))
test_Y_proba = np.amax(rf_model.predict_proba(test_X), axis=1)

print "Writing results to \"%s\"" % output
test_data['predicted_class'] = test_Y
test_data['predicted_class_proba'] = test_Y_proba
test_data.to_csv(output)
