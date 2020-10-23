import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit, learning_curve


import warnings
warnings.filterwarnings("ignore")

sns.set(style="white", context="notebook", palette="deep")

train = pd.read_csv("./titanic/train.csv")
test = pd.read_csv("./titanic/test.csv")
IDtest = test["PassengerId"]

def detect_outliers(df, n, features):

    outlier_indices = []
    for col in features:
        mean = df[col].mean()
        std = df[col].std()
        Q1 = np.percentile( df[col], 25 )
        Q3 = np.percentile( df[col], 75 )
        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers


outlier_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
train.loc[outlier_to_drop, ["Age", "SibSp", "Parch", "Fare"]]

train = train.drop(outlier_to_drop, axis=0).reset_index(drop=True)
train_len = len(train)
dataset = pd.concat(objs = [train, test], axis=0).reset_index(drop=True)

# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()

train.info()
train.isnull().sum()

train.head()
train.dtypes
train.describe()

# 3. Feature analysis
# 3.1 Numerical values

g = sns.heatmap(train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")

g = sns.catplot(x="SibSp", y="Survived", data=train, kind="bar", size=6, palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar")
g.despine(left=True)
g = g.set_ylabels("survival probability")


g = sns.FacetGrid(train, col="Survived")
g = g.map(sns.distplot, "Age")


# Age distribution seems to be a tailed distribution, maybe a gaussian distribution.

# We notice that age distributions are not the same in the survived and not survived subpopulations.
# Indeed, there is a peak corresponding to young passengers, that have survived.
# We also see that passengers between 60-80 have less survived.

# So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.

# It seems that very young passengers have more chance to survvied.

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color = "red", shade=True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], color = "blue", shade=True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived", "Survived"])



# When we superimpose the two densities, we clearly se a peak corresponding (between 0 and 5) to babies and very young childrens.

# Fare

dataset["Fare"].isnull().sum()

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

# Since we have one missing value, I decided to fill it with the median value which will not have an important effect on the prediciton.

# Explore Fare distribution
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")

# As we can see, Fare distribution is very skewed.
# This can lead to overweight very high values in the model, even if it is scaled.
# In this case, it is better to transform it with the log function to reduce this skew.

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i>0 else 0)

g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")

# Skewness is clearly reduced after the log transformation

# 3.2 Categorical values
# Sex

g = sns.barplot(x="Sex", y="Survived", data=train)
g = g.set_ylabel("Survival Probability")

train[["Sex", "Survived"]].groupby("Sex").mean()

# It is clearly obvious that Male have less chance to survive than Female.
# So Sex, might play an important role in the prediction of the survival

# For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation : "Women and children first".


# Pclass

# Explore Pclass vs Survived
g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", size=6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train, size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")

train[["Pclass", "Survived", "Sex"]].groupby(["Pclass", "Sex"]).describe()

# The passenger survival is not the same in the 3 classes.
# First class passengers have more chance to survive than second class and third class passengers.
# This trend is conserved when we look at both male and female passengers.


# Embarked

dataset["Embarked"].isnull().sum()

# Fill Embarked nan values of dataset set with "S" most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")

# Since we have two missing values, I decided to fill them with the most frequent value of "Embarked" (S).

g = sns.catplot(x="Embarked", y="Survived", data=train, size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")


# It seems that passenger coming from Cherbourg (C) have more chance to survive.
# My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).

# Let's see the Pclass distribution vs Embarked

g = sns.catplot(x="Pclass", col="Embarked", data=train, size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Indeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas CHerbourg passengers are mostly in first class which have the highest survival rate.

# At this point, I can't explain why first class has an higher survival rate.
# My hypothesis is that first class passengers were prioritized during the evacuation due to their influence.



# 4. Filling missing values

# 4.1 Age
# As we see, Age column contains 256 missing values in the whole dataset.
# since there is subpopulations that have more chanc eto survived (children for example), it is preferable to keep the age feature and to impoute the missing values.

# To address this problem, I looked at the most correlated features with Age (Sex, Parch, Pclass and SibSP).

g = sns.catplot(y="Age", x="Sex", data=dataset, kind="box")
g = sns.catplot(y="Age", x="Sex", hue="Pclass", data=dataset, kind="box")
g = sns.catplot(y="Age", x="Parch", data=dataset, kind="box")
g = sns.catplot(y="Age", x="SibSp", data=dataset, kind="box")

# Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.
# However, 1st class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
# Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})

g = sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap="BrBG", annot=True)

# The correlation map confirms the factorplots observations except for Parch.
# Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.

# In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.

# So, I decided to use SibSp, Parch and Pclass in order to impute the missing ages.
# The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

# Filling missing value of AGe
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset["SibSp"] == dataset.iloc[i]["SibSp"]) & (dataset["Parch"] == dataset.iloc[i]["Parch"]) & (dataset["SibSp"] == dataset.iloc[i]["SibSp"]))].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else :
        dataset["Age"].iloc[i] = age_med


g = sns.catplot(x="Survived", y="Age", data=train, kind="box")
g = sns.catplot(x="Survived", y="Age", data=train, kind="violin")

# No difference between median value of age in survived and not survived subpopulation.
# But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.

# 5. Feature Engineering.
# 5.1 Name / Title

dataset["Name"].head()


# The Name feature contains information on passenger's title.
# Since some passenger with distingused title may be preferred during the evacuation, it is interesting to add them to the model.

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


g = sns.catplot(x="Title", data=dataset, kind="count")
g = plt.setp(g.get_xticklabels(), rotation=45)


dataset["Title"] = dataset["Title"].replace(["Lady", "the countess", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
dataset["Title"] = dataset["Title"].map({
    "Master":0, "Mist":1, "Ms":1, "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3
})
dataset["Title"] = dataset["Title"].astype(int)



g = sns.countplot(x="Title", data=dataset)
g = sns.catplot(x="Title", data=dataset, kind="count")
g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])


g = sns.catplot(x="Title", y="Survived", data=dataset, kind="bar")
g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
g = g.set_ylabels("survival probability")


# Women and children first
# It is interesting to note that passengers with rare title have more hcance to survive

dataset.drop(labels=["Name"], axis=1, inplace=True)

# 5.2 Family Size
# We can imagin that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation.
# So I choosed to create a "Fize" (family size) feature which is the sum of SibSp, Parch and 1 (including the passenger

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

g = sns.catplot(x="Fsize", y="Survived", data=dataset, kind="point")
g = g.set_ylabels("Survival Probability")


# The family size seems to play an important role, survival probability is worst for large families

# Additionally, I decided to create 4 categories of family size.

