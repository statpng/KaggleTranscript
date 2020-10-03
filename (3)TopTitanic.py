# https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

# Feature analysis
# Feature engineering
# Modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from collections import Counter
from sklearn.ensemble import RandomForestClassifier, \
                             AdaBoostClassifier, \
                             GradientBoostingClassifier, \
                             ExtraTreesClassifier, \
                             VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style="white", context="notebook", palette="deep")

# Load and check data
# Load data

train = pd.read_csv("./titanic/train.csv")
test = pd.read_csv("./titanic/test.csv")
IDtest = test["PassengerId"]

def detect_outliers(df, n, features):
    """
    Take a dataframe df of features and returns a list of the indices corresponding to the observations containing more than n outliers according to the Tukey method
    :param df: dataframe
    :param n: features
    :param features: feature name to be investigated
    :return: outlier_indices
    """
    outlier_indices = []

    # iterate over features (columns)
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers

Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
train.loc[Outliers_to_drop]

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset = dataset.fillna(np.nan)
dataset.isnull().sum()

train.info()
train.isnull().sum()

train.head()

train.dtypes
train.describe()

# Feature Analysis
g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")

g = sns.catplot(x="SibSp", y="Survived", kind="bar", data=train, size=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Parch
g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar", size=6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Age
g = sns.FacetGrid(train, col = "Survived")
g = g.map(sns.distplot, "Age")

g = sns.kdeplot(train["Age"][(train["Survived"]==0) & (train["Age"].notnull())], color = "Red", shade = True )
g = sns.kdeplot(train["Age"][(train["Survived"]==1) & (train["Age"].notnull())], color = "Blue", shade = True )

g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])

# Fare
dataset["Fare"].isnull().sum()
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")

# Categorical values

# Sex
g = sns.barplot(x="Sex", y="Survived", data=train)
g = g.set_ylabel("Survival Probability")

train[["Sex", "Survived"]].groupby("Sex").mean()

# Pclass
g = sns.catplot(x="Pclass", y="Survived", data = train, kind="bar", size = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

g = sns.catplot(x="Pclass", y="Survived", hue="Sex", kind="bar", data=data, size=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Embarked
dataset["Embarked"].isnull().sum()
dataset["Embarked"] = dataset["Embarked"].fillna("S")

g = sns.catplot(x="Embarked", y="Survived", kind="bar", data=train, size=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

g = sns.catplot("Pclass", col="Embarked", kind="count", data=train, size=6, palette="muted")

# Filling missing values

# Age
g = sns.catplot(x = "Sex", y = "Age", kind="box", data=dataset)
g = sns.catplot(x = "Sex", y = "Age", hue="Pclass", kind="box", data=dataset)
g = sns.catplot(x = "Parch", y = "Age", kind="box", data=dataset)
g = sns.catplot(x = "SibSp", y = "Age", kind="box", data=dataset)

dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1}) # male --> 0; female --> 1
g = sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap="BrBG", annot=True)

# Filling missing value of Age

index_NaN_age = list( dataset["Age"][dataset["Age"].isnull()].index )

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset["SibSp"] == dataset.iloc[i]["SibSp"]) &
                               (dataset["Parch"] == dataset.iloc[i]["Parch"]) &
                               (dataset["Pclass"] == dataset.iloc[i]["Pclass"])
                               )].median()
    if not np.isnan(age_pred) :
        dataset["Age"].iloc[i] = age_pred
    else :
        dataset["Age"].iloc[i] = age_med


g = sns.catplot(x="Survived", y="Age", kind="box", data=train)
g = sns.catplot(x="Survived", y="Age", data=train, kind="violin")


# Feature Engineering
dataset["Name"].head()

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()

g = sns.countplot(x="Title", data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)

dataset["Title"] = dataset["Title"].replace(["Lady", "the Countess", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
dataset["Title"] = dataset["Title"].map({"Master":0,
                                         "Miss":1,
                                         "Ms":1,
                                         "Mme":1,
                                         "Mlle":1,
                                         "Mrs":1,
                                         "Mr":2,
                                         "Rare":3 })
dataset["Title"] = dataset["Title"].astype(int)

dataset["Title"].value_counts()

g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])

g = sns.catplot(x="Title", y="Survived", kind="bar", data=dataset)
g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
g = g.set_ylabels("survival probability")

dataset.drop(labels = ["Name"], axis=1, inplace=True)

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

g = sns.catplot(x="Fsize", y="Survived", kind="point", data=dataset)
g = g.set_ylabels("Survival probability")

dataset["Single"] = dataset["Fsize"].map(lambda s: 1 if s == 1 else 0)
dataset["SmallF"] = dataset["Fsize"].map(lambda s: 1 if s == 2 else 0)
dataset["MedF"] = dataset["Fsize"].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset["LargeF"] = dataset["Fsize"].map(lambda s: 1 if s >= 5 else 0)

dataset[["Single", "SmallF", "MedF", "LargeF"]].apply(lambda x: x.value_counts(), axis=0)

fig, ax=plt.subplots(2,2,figsize=(10,10))
sns.barplot(x = "Single", y="Survived", data=dataset, ax=ax[0,0])
ax[0,0].set_ylabel("Survival probability")
g = sns.barplot(x = "SmallF", y="Survived", data=dataset, ax=ax[0,1])
ax[0,1].set_ylabel("Survival probability")
g = sns.barplot(x = "MedF", y="Survived", data=dataset, ax=ax[1,0])
ax[1,0].set_ylabel("Survival probability")
g = sns.barplot(x = "LargeF", y="Survived", data=dataset, ax=ax[1,1])
ax[1,1].set_ylabel("Survival probability")

dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix = "Em")

dataset.head(4)

# Cabin
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()

dataset["Cabin"][dataset["Cabin"].notnull()].head()

dataset["Cabin"] = pd.Series( [i[0] if not pd.isnull(i) else "X" for i in dataset["Cabin"] ])

ord = ["A", "B", "C", "D", "E", "F", "G", "T", "X"]
g = sns.countplot( dataset["Cabin"], order = ord )

g = sns.catplot(x="Cabin", y="Survived", kind="bar", data=dataset, order = ord)
g = g.set_ylabels("Survival Probability")

dataset = pd.get_dummies(dataset, prefix = "Cabin", columns=["Cabin"])

dataset["Ticket"].head()

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(" ")[0])
    else :
        Ticket.append("X")

dataset["Ticket"] = Ticket
dataset["Ticket"].head()

dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix = "T")
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")

dataset.drop(labels = ["PassengerId"], axis=1, inplace=True)

dataset.head()


# Modeling
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"], axis=1, inplace=True)

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"], axis=1)


# Simple modeling

