import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from dateutil import tz

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")
# ----------------------------------------------------------------------------------

DATA_FILE = 'data/data.csv'



def split_data(data_basic):
    # ## 3 - Data Training - basic dataset
    #data_basic = data[['device_height', 'device_width', 'click', 'app_score', 'local_time_naive', 'local_weekday', 'night', 'morning', 'noon', 'afternoon']]

    X = data_basic
    y = X['click']
    click_time = pd.DataFrame(X['local_time_naive'])
    X = X.drop(['click', 'local_time_naive'], axis =1)
    X = pd.get_dummies(X, dummy_na=False, drop_first=True)
    print('split_data - columns: ', X.columns)


    # split to train and test datasets - test is the last month and will be used only for final check so it won't effect the training
    # process
    # Train will be split later to train and test
    max_time = click_time['local_time_naive'].max()
    split_time = max_time - timedelta(days = int(30))
    X_train_valid = X[ click_time['local_time_naive'] < split_time ]
    y_train_valid = y[ click_time['local_time_naive'] < split_time ]
    X_test = X[ click_time['local_time_naive'] >= split_time ]
    y_test = y[ click_time['local_time_naive'] >= split_time ]
    print('X_train_valid: ', X_train_valid.shape)
    print('X_test: ', X_test.shape)


    # Since the dataset is unbalanced, and we have a large dataset,
    # I use the undersample method the balance it using under sample method to balance classes
    # I am using random_state to reproduce the same UnderSample later
    undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=1)

    X_train_valid, y_train_valid = undersampler.fit_resample(X_train_valid, y_train_valid)
    print(np.sum(y_train_valid == 0))
    print(np.sum(y_train_valid == 1))


    # I split the train data to train and validation datasets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2, random_state=2)
    # I am using random_state to reproduce the same spl later
    print('X_train: ', X_train.shape)
    print('X_valid: ', X_valid.shape)

    return X_train_valid, y_train_valid, X_train, X_valid, y_train, y_valid, X_test, y_test


def logistic_regression(X_train_valid, y_train_valid, X_train, y_train, X_valid, y_valid, scaler):
    lr = LogisticRegression()
    lr.fit(scaler.transform(X_train), y_train)

    y_pred = lr.predict(scaler.transform(X_train))
    print('train accuracy: ' + str(round(accuracy_score(y_train, y_pred),2)))

    y_pred_valid = lr.predict(scaler.transform(X_valid))
    print('validation accuracy: ' + str(round(accuracy_score(y_valid, y_pred_valid),2)))


    # i now run with kfold, to see the data is homogeneous, and we get similar results for different train-test splits
    k = 10
    lr = LogisticRegression()
    scores = cross_val_score(lr, scaler.transform(X_train_valid), y_train_valid, cv=k)
    print('scores: ', scores)


def random_forest(X_train, y_train, X_valid, y_valid, scaler):
    # Trying with set of parameters to see the best classifier of this type
    # Running train and validation to see I am not in overfit
    for n_est in [10, 25, 50, 100]:
        for max_dep in [ 4, 8, 15]:
            for min_samp_leaf in [2, 4, 6]:

                rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, min_samples_leaf=min_samp_leaf)

                rf.fit(scaler.transform(X_train), y_train)

                y_pred = rf.predict(scaler.transform(X_train))

                y_pred_valid = rf.predict(scaler.transform(X_valid))

                print(f'{n_est} estimators, depth is {max_dep}, num samples in leaf: {min_samp_leaf}')
                print('--train score = ' + str(round(accuracy_score(y_train, y_pred), 2)) +
                   ' , valid score = ' + str(round(accuracy_score(y_valid, y_pred_valid), 2)))

    # Looks like the set of parameters "25 estimators, depth is 15, num samples in leaf :6" give improvement to 0.61
    # on train set and 0.59 on validation set


def adaboost_classifier(X_train, y_train, X_valid, y_valid, scaler):
    # trying AdaBoostClassifier
    for n_estimators in [10, 15, 50, 100, 200, 1000]:

        ab = AdaBoostClassifier(n_estimators=n_estimators)

        ab.fit(scaler.transform(X_train), y_train)

        y_pred = ab.predict(scaler.transform(X_train))

        y_pred_valid = ab.predict(scaler.transform(X_valid))

        print(f'number of estimators is {n_estimators}')
        print(f'--train score = {round(accuracy_score(y_train, y_pred),2)} , validation score = {round(accuracy_score(y_valid, y_pred_valid),2)}')


    # worse results than those we got with the Random Forest.
    # basic data set summary: we got score result of 0.59 using random forest. Other classifier gave got lower scores


def gradient_boost(X_train, y_train, X_valid, y_valid, scaler):
    for n_estimators in [10, 15, 50, 100]:
        for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:

            gb = GradientBoostingClassifier(n_estimators=n_estimators, max_features=20, learning_rate=lr)

            gb.fit(X_train, y_train)

            y_pred = gb.predict(scaler.transform(X_train))

            y_pred_valid = gb.predict(scaler.transform(X_valid))

            print(f'number of estimators is {n_estimators} , learning rate = {lr}')
            print(f'--train score = {round(accuracy_score(y_train, y_pred),2)} , validation score = {round(accuracy_score(y_valid, y_pred_valid),2)}')


def xg_boost(X_train, y_train, X_valid, y_valid, scaler):
    for n_estimators in [10, 15, 50, 100]:
        for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
            xb = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=lr)
            xb.fit(X_train, y_train)
            y_pred = xb.predict(scaler.transform(X_train))
            y_pred_valid = xb.predict(scaler.transform(X_valid))

            print(f'number of estimators is {n_estimators} , learning rate = {lr}')
            print(f'train score = {round(accuracy_score(y_train, y_pred),2)} , validation score = {round(accuracy_score(y_valid, y_pred_valid),2)}')


def run_models(data):
    print(f'run_models - columns:\n{list(data.columns)}\n')
    X_train_valid, y_train_valid, X_train, X_valid, y_train, y_valid, X_test, y_test = split_data(data)

    # train scaler on the train data
    scaler = StandardScaler()
    scaler.fit(X_train)

    # print('running logistic regression')
    # logistic_regression(X_train_valid, y_train_valid, X_train, y_train, X_valid, y_valid, scaler)

    # results are consistent, meaning data is homogeneous enough and results will not be strongly related to specific train-validation split

    print('\n\nrunning random forest classifier')  # non linear classifier.
    random_forest(X_train, y_train, X_valid, y_valid, scaler)

    print('\n\nrunning adaboost classifier')
    adaboost_classifier(X_train, y_train, X_valid, y_valid, scaler)

    # print('\n\nrunning gradient boost')
    # gradient_boost(X_train, y_train, X_valid, y_valid, scaler)

    # print('\n\nrunning XG boost')
    # xg_boost(X_train, y_train, X_valid, y_valid, scaler)

# -------------------------------------------------------------------


def random_forest_test(n_estimators, max_depth, min_samples_leaf, data):
    #X_train_valid, y_train_valid, X_train, y_train, X_valid, y_valid, X_test, y_test, scaler):

    X_train_valid, y_train_valid, X_train, X_valid, y_train, y_valid, X_test, y_test = split_data(data)

    # train scaler on the train data
    scaler = StandardScaler()
    scaler.fit(X_train)

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    rf.fit(scaler.transform(X_train), y_train)

    y_pred = rf.predict(scaler.transform(X_train))
    y_pred_valid = rf.predict(scaler.transform(X_valid))

    print(f'{n_estimators} estimators, depth is {max_depth} , num samples in leaf : {min_samples_leaf}')
    print(f'train score = {round(accuracy_score(y_train, y_pred),2)} , valid score = {round(accuracy_score(y_valid, y_pred_valid),2)}')


    # analyze precision-recall curve

    # I use the same train-validation-test split as before, so I can compare result with the simple dataset
    # max_time = click_time['local_time_naive'].max()
    # split_time = max_time- timedelta(days = int(30))
    # X_train_valid = X[ click_time['local_time_naive']< split_time ]
    # y_train_valid = y[ click_time['local_time_naive']< split_time ]
    # X_test = X[ click_time['local_time_naive']>= split_time ]
    # y_test = y[ click_time['local_time_naive']>= split_time ]
    # print(X_train_valid.shape)


    y_pred = rf.predict(scaler.transform(X_train_valid))

    print('\n\nconfusion matrix, classification_report on y_train_valid')
    print()
    cm = confusion_matrix(y_true=y_train_valid, y_pred=y_pred, labels=rf.classes_)
    cm = pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_)
    print(cm)

    print(classification_report(y_train_valid, y_pred))



    # 57% from all 'clicks' were classified as 'click'
    # Only 6% of the 'click' classifications were actually 'click' - reletively high "false alarm"


    ### 7 - Last act - check classifier on Test set to see if consistence with the train-data set performance

    print('\nconfusion matrix, classification_report on y_test')
    print()

    y_pred = rf.predict(scaler.transform(X_test))
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=rf.classes_)
    cm = pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_)
    print(cm)


    print(classification_report(y_test, y_pred))

    # result on test set is similar to those we got on the train-validation phase - no overfit, classifier gives
    # consistent results on both sets


def run_all():
    #import feature_ext
    #feature_ext.run()


    # ## RUN TRAIN FLOW
    data = pd.read_csv(DATA_FILE, engine='pyarrow')
    print('data size: ' + str(data.shape[0]) + ' samples , ' + str(data.shape[1]) + ' columns')
    print('sample row:\n', data.sample(1))

    #data = data.sample(500000)


    # run 1
    # data_basic = data[['device_height', 'device_width', 'click', 'app_score', 'local_time_naive', 'local_weekday', 'night', 'morning', 'noon', 'afternoon']]
    # run_models(data_basic)


    # run 2
    # I will try to add more features, using the user_state column, since we saw different user_state has different CR
    # data_basic = data[['user_state', 'device_height', 'device_width', 'click', 'app_score', 'local_time_naive', 'local_weekday', 'night', 'morning', 'noon', 'afternoon']]
    # run_models(data_basic)

    # No improvement when adding the states


    # run 3
    # I will try now the full dataset - basic+states+app_categories


    # ## 5 - Data Training - basic+user_state+app_catagory data set
    # data_full = data
    run_models(data)


    n_estimators = 50
    max_depth = 15
    min_samples_leaf = 2

    random_forest_test(n_estimators, max_depth, min_samples_leaf, data)

    print('\n***********  DONE  ***********\n')


# Logistic Regression give better result for full dataset than the basic and basic+states datasets
# slight improvement by using Random Forest with the full dataset and - results of around 0.62 accuracy
# No improvement using Adaboost



# 6 - Check performance on all train-validation data

run_all()
