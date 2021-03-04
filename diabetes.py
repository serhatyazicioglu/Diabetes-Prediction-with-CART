import warnings
import pydotplus
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from skompiler import skompile
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("datasets/diabetes.csv")

df.info()
df.isnull().sum()

df.describe().T

df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = \
    df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

# Outliers

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.25)
    quartile3 = dataframe[col_name].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

for col in num_cols:
    if col != "Glucose":
        replace_with_thresholds(df, col)

# MISSING VALUES
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_name = missing_values_table(df, na_name=True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_name)

df.pivot_table(df, index=["Outcome"])

for col in df.columns:
    df.loc[(df["Outcome"] == 0) & (df[col].isnull()), col] = df[df["Outcome"] == 0][col].median()
    df.loc[(df["Outcome"] == 1) & (df[col].isnull()), col] = df[df["Outcome"] == 1][col].median()

# FEATURE ENGINEERING

df.loc[(df["Age"] <= 18), "NEW_AGE_CAT"] = "Young"
df.loc[(df["Age"] > 18) & (df["Age"] < 56), "NEW_AGE_CAT"] = "Mature"
df.loc[(df["Age"] > 56), "NEW_AGE_CAT"] = "Old"

df.loc[(df["BMI"] < 18.5), "NEW_BMI_CAT"] = "Underweight"
df.loc[(df["BMI"] > 18.5) & (df["BMI"] < 25), "NEW_BMI_CAT"] = "Normal"
df.loc[(df["BMI"] > 25) & (df["BMI"] < 30), "NEW_BMI_CAT"] = "Overweight"
df.loc[(df["BMI"] > 30) & (df["BMI"] < 40), "NEW_BMI_CAT"] = "Obese"
df.loc[(df["BMI"] > 40), "NEW_BMI_CAT"] = "	Severe Obese"

df["DiaPedFunc_Cat"] = pd.qcut(df["DiabetesPedigreeFunction"], 3, labels=["Low", "Medium", "High"])

df.loc[(df["Glucose"] < 70), "NEW_GLUCOSE_CAT"] = "Low"
df.loc[(df["Glucose"] > 70) & (df["Glucose"] < 99), "NEW_GLUCOSE_CAT"] = "Normal"
df.loc[(df["Glucose"] > 99) & (df["Glucose"] < 126), "NEW_GLUCOSE_CAT"] = "Secret"
df.loc[(df["Glucose"] > 126) & (df["Glucose"] < 200), "NEW_GLUCOSE_CAT"] = "High"

df.loc[df['SkinThickness'] < 30, "NEW_SKIN_THICKNESS"] = "Normal"
df.loc[df['SkinThickness'] >= 30, "NEW_SKIN_THICKNESS"] = "HighFat"

df.loc[df['Pregnancies'] == 0, "NEW_PREGNANCIES"] = "NoPregnancy"
df.loc[((df['Pregnancies'] > 0) & (df['Pregnancies'] <= 4)), "NEW_PREGNANCIES"] = "StdPregnancy"
df.loc[(df['Pregnancies'] > 4), "NEW_PREGNANCIES"] = "OverPregnancy"

df.loc[(df['SkinThickness'] < 30) & (df['BloodPressure'] < 80), "NEW_CIRCULATION_LEVEL"] = "Normal"
df.loc[(df['SkinThickness'] >= 30) & (df['BloodPressure'] >= 80), "NEW_CIRCULATION_LEVEL"] = "CircularAtHighRisk"
df.loc[((df['SkinThickness'] < 30) & (df['BloodPressure'] >= 80))
       | ((df['SkinThickness'] >= 30) & (df['BloodPressure'] < 80)), "NEW_CIRCULATION_LEVEL"] = "CircularAtMediumRisk"

df.loc[(df["BloodPressure"] < 79), "NEW_BLOODPRESSURE_CAT"] = "Normal"
df.loc[(df["BloodPressure"] > 79) & (df["BloodPressure"] < 89), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S1"
df.loc[(df["BloodPressure"] > 89) & (df["BloodPressure"] < 123), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S2"

df["Pre_Age_Cat"] = df["Age"] * df["Pregnancies"]

df["Ins_Glu_Cat"] = df["Glucose"] * df["Insulin"]


def set_insulin(row):
    if 16 <= row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["NEW_INSULIN_CAT"] = df.apply(set_insulin, axis=1)

# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col].astype(str))
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in df.columns:
    label_encoder(df, col)


# ONE-HOT ENCODING

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

one_hot_encoder(df, ohe_cols, drop_first=True)

# Model
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# base model hatası
y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred, y_test))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": [2, 3, 4]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=10, n_jobs=-1, verbose=True)
cart_cv.fit(X_train, y_train)

cart_cv.best_params_

cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train, y_train)

# train hatası
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_model, X_train)
