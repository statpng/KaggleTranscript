# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

train = pd.read_csv("./titanic/train.csv")
test = pd.read_csv("./titanic/test.csv")

PassengerId = test["PassengerId"]
train.head(3)


full_data = [train, test]
train["Name_length"] = train["Name"].apply(len)
test["Name_length"] = test["Name"].apply(len)
train["Has_Cabin"] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test["Has_Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

for dataset in full_data:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

for dataset in full_data:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1

for dataset in full_data:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

for dataset in full_data:
    dataset["Fare"] = dataset["Fare"].fillna(train["Fare"].median())

train["CategoricalFare"] = pd.qcut(train["Fare"], 4)

for dataset in full_data:
    age_avg = dataset["Age"].mean()
    age_std = dataset["Age"].std()
    age_null_count = dataset["Age"].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
    dataset["Age"][np.isnan(dataset["Age"])] = age_null_random_list
    dataset["Age"] = dataset["Age"].astype(int)


train["CategoricalAge"] = pd.cut(train["Age"], 5)

def get_title(name) :
    title_search = re.search(" ([A-Za-z]+)\.", name)
    if title_search :
        return title_search.group(1)
    return ""

get_title(" Mr.ABC")

for dataset in full_data:
    dataset["Title"] = dataset["Name"].apply(get_title)

for dataset in full_data:
    dataset["Title"] = dataset["Title"].replace( ["Lady", "Countess", "Capt", "Col", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

for dataset in full_data:
    dataset["Sex"] = dataset["Sex"].map( {"female": 0, "male": 1} ).astype(int)
    title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

    dataset["Embarked"] = dataset["Embarked"].map( {"S":0, "C":1, "Q":2} ).astype(int)

    dataset.loc[ dataset["Fare"] <= 7.91, "Fare" ] = 0
    dataset.loc[ (dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare" ] = 1
    dataset.loc[ (dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare" ] = 2
    dataset.loc[ (dataset["Fare"] > 31), "Fare" ] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)

    dataset.loc[ dataset["Age"] <= 16, "Age" ] = 0
    dataset.loc[ (dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age" ] = 1
    dataset.loc[ (dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age" ] = 2
    dataset.loc[ (dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age" ] = 3
    dataset.loc[ (dataset["Age"] > 64), "Age" ] = 4


drop_elements = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp"]
train = train.drop(drop_elements, axis=1)
train = train.drop(["CategoricalAge", "CategoricalFare"], axis=1)
test = test.drop(drop_elements, axis=1)


train.head(3)
colormap = plt.cm.RdBu
plt.figure(figsize = (14, 12))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True)
plt.xticks(rotation = 45)

g = sns.pairplot( train[[u"Survived", u"Pclass", u"Sex", u"Age", u"Parch",
                         u"Fare", u"Embarked", u"FamilySize", u"Title"]],
                  hue="Survived", palette="seismic", size=1.2,
                  diag_kind="kde", diag_kws=dict(shade=True),
                  plot_kws=dict(s=10) )
g.set(xticklabels=[])


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, random_state=SEED)


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params["random_state"] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    def predict(self, x):
        return self.clf.predict(x)
    def fit(self, x, y):
        return self.clf.fit(x,y)
    def feature_importances(self, x, y):
        print(self.clf.fit(x,y).feature_importances_)



# Out-of-Fold Predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1,1)



# Generating our Base First-level models
## Random Forest classifier
## Extra Trees classifier
## AdaBoost classifier
## Gradient Boosting classifier
## Support Vector Machine

# Parameters

rf_params = {
    "n_jobs": -1,
    "n_estimators": 500,
    "warm_start": True,
    # "max_features": 0.2,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "verbose": 0
}

et_params = {
    "n_jobs": -1,
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "verbose": 0,

}

ada_params = {
    "n_estimators": 500,
    "learning_rate": 0.75
}

gb_params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_leaf": 2,
    "verbose": 0
}

svc_params = {
    "kernel": "linear",
    "C": 0.025
}

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

y_train = train["Survived"].ravel()
train = train.drop(["Survived"], axis=1)
x_train = train.values
x_test = test.values


# Output of the First level Predictions

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)

print("Training is complete")

rf_feature = rf.feature_importances(x_train, y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train, y_train)

cols = train.columns.values
feature_dataframe = pd.DataFrame(
    {"features": cols,
     "Random Forest feature importances": rf_feature,
     "Extra Trees feature importances": et_feature,
     "AdaBoost feature importances": ada_feature,
     "Gradient Boost feature importances": gb_feature,
     })


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
#     learning_rate = 0.02,
    n_estimators = 2000,
    max_depth = 4,
    min_child_weight = 2,
    gamma = 0.9,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "binary:logistic",
    nthread = -1,
    scale_pos_weight = 1 ).fit(x_train, y_train)
predictions = gbm.predict(x_test)
StackingSubmission = pd.DataFrame({ "PassengerId": PassengerId,
                                    "Survived": predictions})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)


