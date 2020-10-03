# url: https://kaggle-kr.tistory.com/17?category=868316

import numpy as np
import pandas as pd
pd.set_option("display.max_column", 15)
import matplotlib.pyplot as plt
# plt.isinteractive()
# matplotlib.use("TkAgg")
import seaborn as sns
# plt.style.use("seaborn")
# sns.set(font_scale=2.5)

import missingno as msno

import warnings
warnings.filterwarnings("ignore")

# %matplotlib inline

# Import datasets
df_train = pd.read_csv("./titanic/train.csv")
df_test = pd.read_csv("./titanic/test.csv")

df_train.head()
df_train.describe()
df_test.describe()

for col in df_train.columns:
    msg = "column: {:>10}\t Percent of NaN value: {:.2f}%".format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)

for col in df_test.columns:
    msg = "column: {:>10}\t Percent of NaN values: {:.2f}%".format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)

msno.matrix(df=df_train.iloc[:, :], figsize = (8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))

f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train["Survived"].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=ax[0], shadow=False)
ax[0].set_title("Pie plot - Survived")
ax[0].set_ylabel("")
sns.countplot("Survived", data=df_train, ax=ax[1])
ax[1].set_title("Count plot - Survived")

plt.show()


tmp=df_train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=True)
tmp.sum()/tmp.count()

df_train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=True).mean()
pd.crosstab(df_train["Pclass"], df_train["Survived"], margins=True).style.background_gradient(cmap="summer_r")

df_train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=True).mean().sort_values(by="Survived", ascending=False).plot.bar()

y_position = 1.02
f, ax = plt.subplots(1, 2, figsize = (18, 8))
df_train["Pclass"].value_counts().plot.bar(color=["#CD7F32", "#FFDF00", "#D3D3D3"], ax=ax[0])
ax[0].set_title("Number of Passengers By Pclass", y=y_position)
ax[0].set_ylabel("Count")
sns.countplot("Pclass", hue="Survived", data=df_train, ax=ax[1])
ax[1].set_title("Pclass: Survived vs Dead", y=y_position)
plt.show()


# Gender
f, ax = plt.subplots(1, 2, figsize = (20, 10))
df_train[["Sex", "Survived"]].groupby(["Sex"], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title("Survived vs Sex")
sns.countplot("Sex", hue="Survived", data=df_train, ax=ax[1])
ax[1].set_title("Sex: Survived vs Dead", y=y_position)
plt.show()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[["Sex", "Survived"]].groupby(["Sex"], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title("Survived vs Sex")
sns.countplot("Sex", hue="Survived", data=df_train, ax=ax[1])
ax[1].set_title("Sex: Survived vs Dead")
plt.show()


df_train[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
tmp = pd.crosstab(df_train["Sex"], df_train["Survived"], margins=True)
print( tmp )
tmp.style.background_gradient(cmap="summer_r")

sns.catplot("Pclass", "Survived", hue="Sex", data=df_train, size=4, aspect=1.5, kind="point")
sns.catplot(x="Sex", y="Survived", col="Pclass", data=df_train, satureation=0.5, size=4, aspect=1, kind="point")

print("제일 나이 많은 탑승객 : {:.1f} Years".format(df_train["Age"].max()))
print("제일 어린 탑승객 : {:.1f} Years".format(df_train["Age"].min()))
print("탑승객 평균 나이 : {:.1f} Years".format(df_train["Age"].mean()))

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
sns.kdeplot(df_train[df_train["Survived"]==0]["Age"], ax=ax)
plt.legend(["Survived == 1", "Survived == 0"])
plt.show()


df_train["Pclass"].describe()
sns.kdeplot(df_train[df_train["Survived"]==1]["Age"], ax=ax)
pd.crosstab("NULL", df_train["Pclass"])

plt.figure(figsize=(8, 6))
df_train["Age"][df_train["Pclass"]==1].plot(kind="kde")
df_train["Age"][df_train["Pclass"]==2].plot(kind="kde")
df_train["Age"][df_train["Pclass"]==3].plot(kind="kde")

plt.xlabel("Age")
plt.title("Age Distribution within classes")
plt.legend(["1st Class", "2nd Class", "3rd Class"])

cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(df_train[df_train["Age"] < i]["Survived"].sum() / len(df_train[df_train["Age"] < i ]["Survived"]))

def png_list_describe(list):
    out = []
    n = len(list)

    sum = 0; sum2 = 0;
    max_value = 0; min_value = 0;
    for i in range(n):
        sum += list[i]
        sum2 += pow( list[i], 2 )
        if( list[i] > max_value ):
            max_value = list[i]
        if (list[i] < min_value):
            min_value = list[i]

    avg = sum/n;
    variance = sum2/n - pow(avg,2);
    sd = pow(variance, .5)

    out = {"mean":avg, "sd":sd, "min":min_value, "max":max_value}
    for i in out:
        print(i, ":", round(out.get(i), 4))


png_list_describe(cummulate_survival_ratio)

plt.figure(figsize=(7, 7))
plt.plot(cummulate_survival_ratio)
plt.title("Survival rate change depending on range of Age", y=1.02)
plt.ylabel("Survival rate")
plt.xlabel("Range of AGe(0~x")
plt.show()


f, ax = plt.subplots(1, 2, figsize=(18,8))
sns.violinplot("Pclass", "Age", hue="Survived", data=df_train, scale="count", split=True, ax=ax[0])
ax[0].set_title("Pclass and Age vs Survived")
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex", "Age", hue="Survived", data=df_train, scale="count", split=True, ax=ax[1])
ax[1].set_title("Sex and Age vs Survived")
ax[1].set_yticks(range(0,110,10))
# plt.legend(["Survived == 1", "Survived == 0"])
plt.show()
# plt.tight_layout()


# "".join("Embarked")
# " ".join(map(str, ["I", "wnat", 4, "apples" ]))

f, ax = plt.subplots(1, 1, figsize=(7, 7))
df_train[["Embarked", "Survived"]].groupby(["Embarked"], as_index=True).mean().sort_values(by="Survived", ascending=False).plot.bar(ax=ax)
plt.tight_layout()


f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot("Embarked", data=df_train, ax=ax[0,0])
ax[0,0].set_title("(1) No. of Passengers Boarded")
sns.countplot("Embarked", hue="Sex", data=df_train, ax=ax[0,1])
ax[0,1].set_title("(2) Male-Female Split for Embarked")
sns.countplot("Embarked", hue="Survived", data=df_train, ax=ax[1,0])
ax[1,0].set_title("(3) Embarked vs Survived")
sns.countplot("Embarked", hue="Pclass", data=df_train, ax=ax[1,1])
ax[1,1].set_title("(4) Embarked vs Pclass")
plt.subplots_adjust(wspace=.2, hspace=.5)
plt.tight_layout()

type(df_train["Parch"] + 1)
pd.concat( [pd.DataFrame( df_train["Parch"] + 1 ), pd.DataFrame( + df_train["Parch"] + 1 )], axis=1 )

df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1

print("Maximum size of Family: ", df_train["FamilySize"].max())
print("Minimum size of Family: ", df_train["FamilySize"].min())


f, ax = plt.subplots(1, 3, figsize=(40,10))
sns.countplot("FamilySize", data=df_train, ax=ax[0])
ax[0].set_title("(1) No. of Passengers Boarded", y=1.02)

sns.countplot("FamilySize", hue="Survived", data=df_train, ax=ax[1])
ax[1].set_title("(2) Survived countplot depending on FamilySize", y=1.02)

df_train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=True).mean().sort_values(by="Survived", ascending=True).plot.bar(ax=ax[2])
ax[2].set_title("(3) Survived rate depending on FamilySize", y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()



fig, ax=plt.subplots(1, 1, figsize=(8,8))
g = sns.distplot(df_train["Fare"], color="b", label="Skewness : {:.2f}".format(df_train["Fare"].skew()), ax=ax)
g = g.legend(loc="best")

df_test.loc[df_test.Fare.isnull(), "Fare"] = df_test["Fare"].mean()
df_train["Fare"] = df_train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
df_test["Fare"] = df_test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train["Fare"], color="b", label="Skewness : {:.2f}".format(df_train["Fare"].skew()), ax=ax)
g = g.legend(loc="best")

df_train.head()

df_train["Ticket"].value_counts()
