import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=2.2)
plt.style.use("seaborn")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit

from sklearn.metrics import f1_score
import itertools
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
import shap
from tqdm import tqdm
import featuretools as ft
import warnings
warnings.filterwarnings("ignore")
import time

df_train = pd.read_csv("./costa-rican-household-poverty-prediction/train.csv")
df_test = pd.read_csv("./costa-rican-household-poverty-prediction/test.csv")


print("df_train shape:", df_train.shape, "  ", "df_test shape: ", df_test.shape)

df_train.head()

df_train.describe()

df_test.head()


# 1.2 Make description df
description = pd.read_csv("./costa-rican-household-poverty-prediction/codebook.csv")

description


# 1.3 Check null data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = 100 * (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_df = pd.concat( [total, percent], axis=1, keys=["Total", "Percent"])

missing_df.head(20)


# 1.4 Fill missing values
# Below cell is from this kernel:

# if education is "yes" and person is head of household, fill with escolari
df_train.loc[(df_train["edjefa"] == "yes") & (df_train["parentesco1"] == 1), "edjefa"] = df_train.loc[ (df_train["edjefa"] == "yes") & (df_train["parentesco1"] == 1), "escolari"]
df_train.loc[(df_train["edjefe"] == "yes") & (df_train["parentesco1"] == 1), "edjefe"] = df_train.loc[ (df_train["edjefe"] == "yes") & (df_train["parentesco1"] == 1), "escolari"]

df_test.loc[(df_test["edjefa"] == "yes") & (df_test["parentesco1"] == 1), "edjefa"] = df_test.loc[ (df_test["edjefa"] == "yes") & (df_test["parentesco1"] == 1), "escolari"]
df_test.loc[(df_test["edjefe"] == "yes") & (df_test["parentesco1"] == 1), "edjefe"] = df_test.loc[ (df_test["edjefe"] == "yes") & (df_test["parentesco1"] == 1), "escolari"]

# This field is supposed to be interaction between gender and escolari, but it isn't clear what "yes" means, let's fill it with 4
df_train.loc[df_train["edjefa"] == "yes", "edjefa"] = 4
df_train.loc[df_train["edjefe"] == "yes", "edjefe"] = 4

df_test.loc[df_test["edjefa"] == "yes", "edjefa"] = 4
df_test.loc[df_test["edjefe"] == "yes", "edjefe"] = 4

# create feature with max education of either head of household
df_train["edjef"] = np.max(df_train[["edjefa", "edjefe"]], axis=1)
df_test["edjef"] = np.max(df_test[["edjefa", "edjefe"]], axis=1)

# fix some inconsistencies in the data - some rows indicate both that the household does and does not have a toilet,
# if there is no water we'll assume they do not
df_train.loc[(df_train.v14a == 1) & (df_train.sanitario1 == 1) & (df_train.abastaguano == 0), "v14a"] = 0
df_train.loc[(df_train.v14a == 1) & (df_train.sanitario1 == 1) & (df_train.abastaguano == 0), "sanitario1"] = 0

df_test.loc[(df_test.v14a == 1 ) & (df_test.sanitario1 == 1) & (df_test.abastaguano == 0), "v14a"] = 0
df_test.loc[(df_test.v14a == 1 ) & (df_test.sanitario1 == 1) & (df_test.abastaguano == 0), "sanitario1"] = 0


# rez_esz, SQBmeaned
# - rez_esc : Years behind in school -> filled with 0
# - SQBmeaned : square of the mean years of education of adults (>=18) in the household agesq, Age squared -> same with rez_esc -> filled with 0

df_train["rez_esc"].fillna(0, inplace=True)
df_test["rez_esc"].fillna(0, inplace=True)

df_train["SQBmeaned"].fillna(0, inplace=True)
df_test["SQBmeaned"].fillna(0, inplace=True)


# meaneduc
# - meaneduc: average years of education for adults (18+) -> filled with 0

df_train["meaneduc"].fillna(0, inplace=True)
df_test["meaneduc"].fillna(0, inplace=True)


# v18q1
# - v18q1: number of tablets household owns -> if v18q (Do you own a tablet?) == 1, there are some values. If not, only NaN values in v18q1. See below 3 cells.

df_train["v18q"].value_counts()

df_train.loc[df_train["v18q"] == 1, "v18q1"].value_counts()
df_train.loc[df_train["v18q"] == 0, "v18q1"].value_counts()

df_train["v18q1"].fillna(0, inplace=True)
df_test["v18q1"].fillna(0, inplace=True)


# - v2a1 : number of tablets household owns -> if tipovivi3 (rented?) == 1, there are some values. If not, there are also some values.
# - NaN value could be replaced by 0.

df_train["tipovivi3"].value_counts()


sns.kdeplot(df_train.loc[df_train["tipovivi3"] == 1, "v2a1"], label = "Monthly rent payment of household(rented=1")
sns.kdeplot(df_train.loc[df_train["tipovivi3"] == 0, "v2a1"], label = "Monthly rent payment of household(rented=0")
plt.xscale("log")
plt.show()


df_train["v2a1"].fillna(0, inplace=True)
df_test["v2a1"].fillna(0, inplace=True)


total = df_train.isnull().sum().sort_values(ascending=False)
percent = 100 * (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_df = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])

missing_df.head(20)



total = df_test.isnull().sum().sort_values(ascending=False)
percent = 100 * (df_test.isnull().sum() / df_test.isnull().count()).sort_values(ascending=False)
missing_df = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])

missing_df.head(20)


# For now, there are no NaN values.


# 2. Feature engineering
# 2.1 Object features

features_object = [col for col in df_train.columns if df_train[col].dtype == "object"]

features_object



# dependency

# some dependencies are NA, fill those with the square root of the square

df_train["dependency"] = np.sqrt(df_train["SQBdependency"])
df_test["dependency"] = np.sqrt(df_test["SQBdependency"])


# edjefe
# - edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# - replace yes -> 1 and no -> 0

def replace_edjefe(x):
    if x == "yes":
        return 1
    elif x == "no":
        return 0
    else:
        return x

df_train["edjefe"] = df_train["edjefe"].apply(replace_edjefe).astype(float)
df_test["edjefe"] = df_test["edjefe"].apply(replace_edjefe).astype(float)


# edjefa
# - edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# - replace yes -> 1 and no -> 0

def replace_edjefa(x):
    if x == "yes":
        return 1
    elif x == "no":
        return 0
    else:
        return x

df_train["edjefa"] = df_train["edjefa"].apply(replace_edjefa).astype(float)
df_test["edjefa"] = df_test["edjefa"].apply(replace_edjefa).astype(float)


# create feature with max education of either head of household
df_train["edjef"] = np.max(df_train[["edjefa", "edjefe"]], axis=1)
df_test["edjef"] = np.max(df_test[["edjefa", "edjefe"]], axis=1)


# roof and electricity
# I refered to  https://www.kaggle.com/mineshjethva/exploratory-data-analysis-lightgbm.
# Thanks!

df_train["roof_waste_material"] = np.nan
df_test["roof_waste_material"] = np.nan
df_train["electricity_other"] = np.nan
df_test["electricity_other"] = np.nan


def fill_roof_exception(x):
    if (x["techozinc"] == 0) and (x["techoentrepiso"] == 0) and (x["techocane"] == 0) and (x["techootro"] == 0):
        return 1
    else:
        return 0

def fill_no_electricity(x):
    if (x["public"] == 0) and (x["planpri"] == 0) and (x["noelec"] == 0) and (x["coopele"] == 0):
        return 1
    else:
        return 0


df_train["roof_waste_material"] = df_train.apply(lambda x : fill_roof_exception(x), axis=1)
df_test["roof_waste_material"] = df_test.apply(lambda x : fill_roof_exception(x), axis=1)

df_train["electricity_other"] = df_train.apply( lambda x: fill_no_electricity(x), axis=1)
df_test["electricity_other"] = df_test.apply( lambda x: fill_no_electricity(x), axis=1)


# 2.2 Extract cat features
# - According to data description, there are many binary category features.

binary_cat_features = [col for col in df_train.columns if df_train[col].value_counts().shape[0] == 2 ]


# 2.3 Make new features using continuous feature

continuous_features = [col for col in df_train.columns if col not in binary_cat_features]
continuous_features = [col for col in df_train.columns if col not in features_object]
continuous_features = [col for col in df_train.columns if col not in ["Id", "Target", "idhogar"]]


print("There are {} continuous features".format(len(continuous_features)))
for col in continuous_features:
    print("{}: {}".format(col, description.loc[description["Variable name"] == col, "Variable description"].values))

# - hhsize : household size
# - tamhog : size of the household
# what is different?

# - As you can see, the meaning of two features are same but the exact number are different. Are they different?
# - I don't know For now, I decided to drop one feature "tamhog".

df_train["edjef"].value_counts()

df_train.drop("tamhog", axis=1, inplace=True)
df_test.drop("tamhog", axis=1, inplace=True)


# Squared features
# - There are many squared features.
# - Actually, tree models like lightgbm don't need them. But at this kernel, I want to use lightgbm as feature filter model and set entity-embedding as classifier.
# So, let's keep them.

# Family features
# - hogar_nin, hogar_adul, hogar_mayor, hogar_total, r4h1, r4h2, r4h3, r4m1, r4m2, r4m3, r4t1, r4t2, r4t3, tmbhog, tamvid, rez_esc, escolari

# - Family size features (substract, ratio)

df_train["adult"] = df_train["hogar_adul"] - df_train["hogar_mayor"]
df_train["dependency_count"] = df_train["hogar_nin"] + df_train["hogar_mayor"]
df_train["dependency"] = df_train["dependency_count"] / df_train["adult"]
df_train["child_percent"] = df_train["hogar_nin"] / df_train["hogar_total"]
df_train["elder_percent"] = df_train["hogar_mayor"] / df_train["hogar_total"]
df_train["adult_percent"] = df_train["hogar_adul"] / df_train["hogar_total"]
df_train["males_younger_12_years_percent"] = df_train["r4h1"] / df_train["hogar_total"]
df_train["males_older_12_years_percent"] = df_train["r4h2"] / df_train["hogar_total"]
df_train["males_percent"] = df_train["r4h3"] / df_train["hogar_total"]
df_train["females_younger_12_years_percent"] = df_train["r4m1"] / df_train["hogar_total"]
df_train["females_older_12_years_percent"] = df_train["r4m2"] / df_train["hogar_total"]
df_train["females_percent"] = df_train["r4m3"] / df_train["hogar_total"]
df_train["persons_younger_12_years_percent"] = df_train["r4t1"] / df_train["hogar_total"]
df_train["persons_older_12_years_percent"] = df_train["r4t2"] / df_train["hogar_total"]
df_train["persons_percent"] = df_train["r4t3"] / df_train["hogar_total"]



df_test["adult"] = df_test["hogar_adul"] - df_test["hogar_mayor"]
df_test["dependency_count"] = df_test["hogar_nin"] + df_test["hogar_mayor"]
df_test["dependency"] = df_test["dependency_count"] / df_test["adult"]
df_test["child_percent"] = df_test["hogar_nin"] / df_test["hogar_total"]
df_test["elder_percent"] = df_test["hogar_mayor"] / df_test["hogar_total"]
df_test["adult_percent"] = df_test["hogar_adul"] / df_test["hogar_total"]
df_test["males_younger_12_years_percent"] = df_test["r4h1"] / df_test["hogar_total"]
df_test["males_older_12_years_percent"] = df_test["r4h2"] / df_test["hogar_total"]
df_test["males_percent"] = df_test["r4h3"] / df_test["hogar_total"]
df_test["females_younger_12_years_percent"] = df_test["r4m1"] / df_test["hogar_total"]
df_test["females_older_12_years_percent"] = df_test["r4m2"] / df_test["hogar_total"]
df_test["females_percent"] = df_test["r4m3"] / df_test["hogar_total"]
df_test["persons_younger_12_years_percent"] = df_test["r4t1"] / df_test["hogar_total"]
df_test["persons_older_12_years_percent"] = df_test["r4t2"] / df_test["hogar_total"]
df_test["persons_percent"] = df_test["r4t3"] / df_test["hogar_total"]



df_train['males_younger_12_years_in_household_size'] = df_train['r4h1'] / df_train['hhsize']
df_train['males_older_12_years_in_household_size'] = df_train['r4h2'] / df_train['hhsize']
df_train['males_in_household_size'] = df_train['r4h3'] / df_train['hhsize']
df_train['females_younger_12_years_in_household_size'] = df_train['r4m1'] / df_train['hhsize']
df_train['females_older_12_years_in_household_size'] = df_train['r4m2'] / df_train['hhsize']
df_train['females_in_household_size'] = df_train['r4m3'] / df_train['hogar_total']
df_train['persons_younger_12_years_in_household_size'] = df_train['r4t1'] / df_train['hhsize']
df_train['persons_older_12_years_in_household_size'] = df_train['r4t2'] / df_train['hhsize']
df_train['persons_in_household_size'] = df_train['r4t3'] / df_train['hhsize']


df_test['males_younger_12_years_in_household_size'] = df_test['r4h1'] / df_test['hhsize']
df_test['males_older_12_years_in_household_size'] = df_test['r4h2'] / df_test['hhsize']
df_test['males_in_household_size'] = df_test['r4h3'] / df_test['hhsize']
df_test['females_younger_12_years_in_household_size'] = df_test['r4m1'] / df_test['hhsize']
df_test['females_older_12_years_in_household_size'] = df_test['r4m2'] / df_test['hhsize']
df_test['females_in_household_size'] = df_test['r4m3'] / df_test['hogar_total']
df_test['persons_younger_12_years_in_household_size'] = df_test['r4t1'] / df_test['hhsize']
df_test['persons_older_12_years_in_household_size'] = df_test['r4t2'] / df_test['hhsize']
df_test['persons_in_household_size'] = df_test['r4t3'] / df_test['hhsize']



df_train["overcrowding_room_and_bedroom"] = (df_train["hacdor"] + df_train["hacapo"]) / 2
df_test["overcrowding_room_and_bedroom"] = (df_test["hacdor"] + df_test["hacapo"]) / 2


df_train['escolari_age'] = df_train['escolari']/df_train['age']
df_test['escolari_age'] = df_test['escolari']/df_test['age']

df_train['age_12_19'] = df_train['hogar_nin'] - df_train['r4t1']
df_test['age_12_19'] = df_test['hogar_nin'] - df_test['r4t1']


df_train['phones-per-capita'] = df_train['qmobilephone'] / df_train['tamviv']
df_train['tablets-per-capita'] = df_train['v18q1'] / df_train['tamviv']
df_train['rooms-per-capita'] = df_train['rooms'] / df_train['tamviv']
df_train['rent-per-capita'] = df_train['v2a1'] / df_train['tamviv']



df_test['phones-per-capita'] = df_test['qmobilephone'] / df_test['tamviv']
df_test['tablets-per-capita'] = df_test['v18q1'] / df_test['tamviv']
df_test['rooms-per-capita'] = df_test['rooms'] / df_test['tamviv']
df_test['rent-per-capita'] = df_test['v2a1'] / df_test['tamviv']



# - You can see that "Total persons in the household" != "# of total individuals in the household".
# - Somewhat weired. But for now I will keep it.


(df_train["hogar_total"] == df_train["r4t3"]).sum()


# Rent per family features
# - I will reduce the number of features using shap, so let's generate many features!!
# Hope catch some fortune features :)


family_size_features = ["adult", "hogar_adul", "hogar_mayor", "hogar_nin", "hogar_total",
                        "r4h1", "r4h2", "r4h3",
                        "r4m1", "r4m2", "r4m3",
                        "r4t1", "r4t2", "r4t3", "hhsize"]

new_feats = []
for col in family_size_features:
    new_col_name = "new_{}_per_{}".format("v2a1", col)
    new_feats.append(new_col_name)
    df_train[new_col_name] = df_train["v2a1"] / df_train[col]
    df_test[new_col_name] = df_test["v2a1"] / df_test[col]


# Ratio feature can have infinite values. So let them be filled with 0.

for col in new_feats:
    df_train[col].replace([np.inf], np.nan, inplace=True)
    df_train[col].fillna(0, inplace=True)

    df_test[col].replace([np.inf], np.nan, inplace=True)
    df_test[col].fillna(0, inplace=True)



# Room per family features
new_feats = []
for col in family_size_features:
    new_col_name = "new_{}_per_{}".format("rooms", col)
    new_feats.append(new_col_name)
    df_train[new_col_name] = df_train["rooms"] / df_train[col]
    df_test[new_col_name] = df_test["rooms"] / df_test[col]


for col in new_feats:
    df_train[col].replace([np.inf], np.nan, inplace=True)
    df_train[col].fillna(0, inplace=True)

    df_test[col].replace([np.inf], np.nan, inplace=True)
    df_test[col].fillna(0, inplace=True)


# BedRoom per family features
new_feats = []
for col in family_size_features:
    new_col_name = "new_{}_per_{}".format("bedrooms", col)
    new_feats.append(new_col_name)
    df_train[new_col_name] = df_train["bedrooms"] / df_train[col]
    df_test[new_col_name] = df_test["bedrooms"] / df_test[col]

for col in new_feats:
    df_train[col].replace(np.inf, np.nan, inplace=True)
    df_train[col].fillna(0, inplace=True)

    df_test[col].replace(np.inf, np.nan, inplace=True)
    df_test[col].fillna(0, inplace=True)


print(df_train.shape, df_test.shape)

new_feats = []
for col in family_size_features:
    new_col_name = "new_{}_per_{}".format("v18q1", col)
    new_feats.append(new_col_name)
    df_train[new_col_name] = df_train["v18q1"] / df_train[col]
    df_test[new_col_name] = df_test["v18q1"] / df_test[col]

for col in new_feats:
    df_train[col].replace([np.inf], np.nan, inplace=True)
    df_train[col].fillna(0, inplace=True)

    df_test[col].replace([np.inf], np.nan, inplace=True)
    df_test[col].fillna(0, inplace=True)



# Phone per family features
new_feats = []
for col in family_size_features:
    new_col_name = "new_{}_per_{}".format("qmobilephone", col)
    new_feats.append(new_col_name)
    df_train[new_col_name] = df_train["qmobilephone"] / df_train[col]
    df_test[new_col_name] = df_train["qmobilephone"] / df_test[col]

for col in new_feats:
    df_train[col].replace([np.inf], np.nan, inplace=True)
    df_train[col].fillna(0, inplace=True)

    df_test[col].replace([np.inf], np.nan, inplace=True)
    df_test[col].fillna(0, inplace=True)


# rez_esc (Years behind in school) per family features

new_feats = []
for col in family_size_features:
    new_col_name = "new_{}_per_{}".format("rez_esc", col)
    new_feats.append(new_col_name)
    df_train[new_col_name] = df_train["rez_esc"] / df_train[col]
    df_test[new_col_name] = df_train["rez_esc"] / df_train[col]

for col in new_feats:
    df_train[col].replace([np.inf], np.nan, inplace=True)
    df_train[col].fillna(0, inplace=True)

    df_test[col].replace([np.inf], np.nan, inplace=True)
    df_test[col].fillna(0, inplace=True)


df_train["rez_esc_age"] = df_train["rez_esc"] / df_train["age"]
df_train["rez_esc_escolari"] = df_train["rez_esc"] / df_train["escolari"]

df_test["rez_esc_age"] = df_test["rez_esc"] / df_test["age"]
df_test["rez_esc_escolari"] = df_test["rez_esc"] / df_test["escolari"]



# Rich features
# I think the more richer, the larger number of phones and tabulets

df_train["tabulet_x_qmobilephone"] = df_train["v18q1"] * df_train["qmobilephone"]
df_test["tabulet_x_qmobilephone"] = df_test["v18q1"] * df_test["qmobilephone"]


# wall, roof, floor may be key factor
# Let's multiply each of them. Because they are binary cat features, so multiplification of each feature generates new categorical features.

# wall and roof
for col1 in ["epared1", "epared2", "epared3"]:
    for col2 in ["etecho1", "etecho2", "etecho3"]:
        new_col_name = "new_{}_x_{}".format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]


for col1 in ["epared1", "epared2", "epared3"]:
    for col2 in ["eviv1", "eviv2", "eviv3"]:
        new_col_name = "new_{}_x_{}".format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

# roof and floor
for col1 in ["etecho1", "etecho2", "etecho3"]:
    for col2 in ["eviv1", "eviv2", "eviv3"]:
        new_col_name = "new_{}_x_{}".format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]



# combination using three features

for col1 in ["epared1", "epared2", "epared3"]:
    for col2 in ["etecho1", "etecho2", "etecho3"]:
        for col3 in ["eviv1", "eviv2", "eviv3"]:
            new_col_name = "new_{}_x_{}_x_{}".format(col1, col2, col3)
            df_train[new_col_name] = df_train[col1] * df_train[col2] * df_train[col3]
            df_test[new_col_name] = df_test[col1] * df_test[col2] * df_test[col3]



print(df_train.shape, df_test.shape)

# I want to mix electricity and energy features -> energy features

for col1 in ["public", "planpri", "noelec", "coopele"]:
    for col2 in ["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"]:
        new_col_name = "new_{}_x_{}".format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]


# I want to mix toilet and rubbish disposal features -> other_infra features

for col1 in ["sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"]:
    for col2 in ["elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6"]:
        new_col_name = "new_{}_x_{}".format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

# I want to mix toilet and water provision features -> water features

for col1 in ["abastaguadentro", "abastaguafuera", "abastaguano"]:
    for col2 in ["sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"]:
        new_col_name = "new_{}_x_{}".format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

print(df_train.shape, df_test.shape)




# I want to mix education and area features -> education_zone_features

for col1 in ['area1', 'area2']:
    for col2 in ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']:
        new_col_name = 'new_{}_x_{}'.format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

# Mix region and education

for col1 in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:
    for col2 in ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']:
        new_col_name = 'new_{}_x_{}'.format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

print(df_train.shape, df_test.shape)


# Multiply television / mobilephone / computer / tabulet / refrigerator -> electronics features

df_train["electronics"] = df_train["computer"] * df_train["mobilephone"] * df_train["television"] * df_train["v18q"] * df_train["refrig"]
df_test["electronics"] = df_test["computer"] * df_test["mobilephone"] * df_test["television"] * df_test["v18q"] * df_test["refrig"]

df_train["no_appliances"] = df_train["refrig"] + df_train["computer"] + df_train["television"] + df_train["mobilephone"]
df_test["no_appliances"] = df_test["refrig"] + df_test["computer"] + df_test["television"] + df_test["mobilephone"]


# Mix wall materials of roof, floor, wall

for col1 in ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras',
             'paredother']:
    for col2 in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:
        new_col_name = 'new_{}_x_{}'.format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

for col1 in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:
    for col1 in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:
        new_col_name = 'new_{}_x_{}'.format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

for col1 in ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras',
             'paredother']:
    for col2 in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:
        new_col_name = 'new_{}_x_{}'.format(col1, col2)
        df_train[new_col_name] = df_train[col1] * df_train[col2]
        df_test[new_col_name] = df_test[col1] * df_test[col2]

for col1 in ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras',
             'paredother']:
    for col2 in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:
        for col3 in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:
            new_col_name = 'new_{}_x_{}_x_{}'.format(col1, col2, col3)
            df_train[new_col_name] = df_train[col1] * df_train[col2] * df_train[col3]
            df_test[new_col_name] = df_test[col1] * df_test[col2] * df_train[col3]


print(df_train.shape, df_test.shape)

# Wow without any aggregation features, we have 446 features for now.
# Actually, mixing the materials of wall make thousands of features.
# I don't want to do that because of computational costs!

# Remove features with only one value.

cols_with_only_one_value = []
for col in df_train.columns:
    if col == "Target":
        continue
    if df_train[col].value_counts().shape[0] == 1 or df_test[col].value_counts().shape[0] == 1:
        print(col)
        cols_with_only_one_value.append(col)


# Let's remove them!

df_train.drop(cols_with_only_one_value, axis=1, inplace=True)
df_test.drop(cols_with_only_one_value, axis=1, inplace=True)



# Check whether both train and test have same features

cols_train = np.array(sorted([col for col in df_train.columns if col != "Target"]))
cols_test = np.array(sorted([col for col in df_test.columns if col != "Target"]))

(cols_train == cols_test).sum() == len(cols_train)

# Good, let's move!

# 2.4 Aggregating features

# In this competition, each sample are member of specific household (idhogar).
# So let's aggregate based on "idhogar" values.

# Aggregation for family features.

def max_min(x):
    return x.max() - x.min()

agg_train = pd.DataFrame()
agg_test = pd.DataFrame()

for item in tqdm(family_size_features):
    for i, function in enumerate(["mean", "std", "min", "max", "sum", "count", max_min]):
        group_train = df_train[item].groupby(df_train["idhogar"]).agg(function)
        group_test = df_test[item].groupby(df_test["idhogar"]).agg(function)
        if i == 6:
            new_col = item + "_new_" + "max_min"
        else:
            new_col = item + "_new_" + function

        agg_train[new_col] = group_train
        agg_test[new_col] = group_test

print("new aggregated train set has {} rows and {} features".format(agg_train.shape[0], agg_train.shape[1]))
print("new aggregated test set has {} rows and {} features".format(agg_test.shape[0], agg_test.shape[1]))



aggr_list = ["rez_esc", "dis", "male", "female",
             "estadocivil1", "estadocivil2", "estadocivil3", "estadocivil4", "estadocivil5", "estadocivil6", "estadocivil7",
             "parentesco2", "parentesco3", "parentesco4", "parentesco5", "parentesco6", "parentesco7", "parentesco8", "parentesco9", "parentesco10", "parentesco11", "parentesco12",
             "instlevel1", "instlevel2", "instlevel3", "instlevel4", "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9",
             "epared1", "epared2", "epared3",
             "etecho1", "etecho2", "etecho3",
             "eviv1", "eviv2", "eviv3",
             "refrig", "television", "mobilephone", "area1", "area2", "v18q", "edjef"]


for item in tqdm(aggr_list):
    for function in ["count", "sum"]:
        group_train = df_train[item].groupby(df_train["idhogar"]).agg(function)
        group_test = df_test[item].groupby(df_test["idhogar"]).agg(function)

        new_col = item + "_new1_" + function
        agg_train[new_col] = group_train
        agg_test[new_col] = group_test


print("new aggregated train set has {} rows and {} features".format(agg_train.shape[0], agg_train.shape[1]))
print("new aggregated test set has {} rows and {} features".format(agg_test.shape[0], agg_test.shape[1]))





aggr_list = ['escolari', 'age', 'escolari_age', 'dependency', 'bedrooms', 'overcrowding', 'rooms', 'qmobilephone', 'v18q1']

for item in tqdm(aggr_list):
    for function in ['mean','std','min','max','sum', 'count', max_min]:
        group_train = df_train[item].groupby(df_train['idhogar']).agg(function)
        group_test = df_test[item].groupby(df_test['idhogar']).agg(function)
        if i == 6:
            new_col = item + '_new2_' + 'max_min'
        else:
            new_col = item + '_new2_' + function
        agg_train[new_col] = group_train
        agg_test[new_col] = group_test

print('new aggregate train set has {} rows, and {} features'.format(agg_train.shape[0], agg_train.shape[1]))
print('new aggregate test set has {} rows, and {} features'.format(agg_test.shape[0], agg_test.shape[1]))



agg_train = agg_train.reset_index()
agg_test = agg_test.reset_index()

train_agg = pd.merge(df_train, agg_train, on="idhogar")
test_agg = pd.merge(df_test, agg_test, on="idhogar")



aggr_list = ['rez_esc', 'dis', 'male', 'female',
              'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7',
              'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10',
              'parentesco11', 'parentesco12',
              'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
             'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'refrig', 'television', 'mobilephone',
            'area1', 'area2', 'v18q', 'edjef']

for lugar in ["lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6"]:
    group_train = df_train[[lugar, "idhogar"] + aggr_list].groupby([lugar, "idhogar"]).sum().reset_index()
    group_train.columns = [lugar, "idhogar"] + ["new3_{}_idhogar_{}".format(lugar, col) for col in group_train][2:]

    group_test = df_test[[lugar, "idhogar"] + aggr_list].groupby([lugar, "idhogar"]).sum().reset_index()
    group_test.columns = [lugar, "idhogar"] + ["new3_{}_idhogar_{}".format(lugar, col) for col in group_test][2:]

    train_agg = pd.merge(train_agg, group_train, on=[lugar, "idhogar"])
    test_agg = pd.merge(test_agg, group_test, on=[lugar, "idhogar"])


print('train shape:', train_agg.shape, 'test shape:', test_agg.shape)




aggr_list = ['rez_esc', 'dis', 'male', 'female',
             'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6',
             'estadocivil7',
             'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8',
             'parentesco9', 'parentesco10',
             'parentesco11', 'parentesco12',
             'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7',
             'instlevel8', 'instlevel9',
             'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'refrig',
             'television', 'mobilephone',
             'area1', 'area2', 'v18q', 'edjef']

for lugar in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:
    group_train = df_train[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).sum().reset_index()
    group_train.columns = [lugar, 'idhogar'] + ['new4_{}_idhogar_{}'.format(lugar, col) for col in group_train][2:]

    group_test = df_test[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).sum().reset_index()
    group_test.columns = [lugar, 'idhogar'] + ['new4_{}_idhogar_{}'.format(lugar, col) for col in group_test][2:]

    train_agg = pd.merge(train_agg, group_train, on=[lugar, 'idhogar'])
    test_agg = pd.merge(test_agg, group_test, on=[lugar, 'idhogar'])

print('train shape:', train_agg.shape, 'test shape:', test_agg.shape)







cols_nums = ['age', 'meaneduc', 'dependency',
             'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',
             'bedrooms', 'overcrowding']

for function in tqdm(['mean', 'std', 'min', 'max', 'sum', 'count', max_min]):
    for lugar in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:
        group_train = df_train[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).agg(function).reset_index()
        group_train.columns = [lugar, 'idhogar'] + ['new5_{}_idhogar_{}_{}'.format(lugar, col, function) for col in
                                                    group_train][2:]

        group_test = df_test[[lugar, 'idhogar'] + aggr_list].groupby([lugar, 'idhogar']).agg(function).reset_index()
        group_test.columns = [lugar, 'idhogar'] + ['new5_{}_idhogar_{}_{}'.format(lugar, col, function) for col in
                                                   group_test][2:]

        train_agg = pd.merge(train_agg, group_train, on=[lugar, 'idhogar'])
        test_agg = pd.merge(test_agg, group_test, on=[lugar, 'idhogar'])

print('train shape:', train_agg.shape, 'test shape:', test_agg.shape)



# - According to data description, ONLY the heads of household are used in scoring.
# - All household members are included in test + the sample submission, but only heads of households are scored.


train = train_agg.query("parentesco1==1")
test = test_agg.query("parentesco1==1")

train["dependency"].replace(np.inf, 0, inplace=True)
test["dependency"].replace(np.inf, 0, inplace=True)

submission = test[["Id"]]

# Remove useless features to reduce dimension
train.drop(columns=['idhogar','Id', 'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)
test.drop(columns=['idhogar','Id',  'agesq', 'hogar_adul', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned'], inplace=True)


correlation = train.corr()
correlation = correlation['Target'].sort_values(ascending=False)

print("final_data size", train.shape, test.shape)

print(f"The most 20 positive feature: \n{correlation.head(40)}")
print(f"The most 20 negatives feature: \n{correlation.tail(20)}")


# Feature selection using shap

binary_cat_features = [col for col in train.columns if train[col].value_counts().shape[0] == 2]
object_features = ["edjefe", "edjefa"]

categorical_feats = binary_cat_features + object_features


def evaluate_macroF1_lgb(truth, predictions):
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average = "macro")
    return ("macroF1", f1, True)



y = train["Target"]
train.drop(columns=["Target"], inplace=True)

def print_execution_time(start):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("*"*20, "Execution ended in {:0>2}h {:0>2)m {:05.2f}s".format(int(hours), int(minutes), seconds), "*"*20)


def extract_good_features_using_shap_LGB(params, SEED):
    clf = lgb.LGBMClassifier(objective = "multiclass",
                             random_state = 1989,
                             max_depth = params["max_depth"],

                             )
