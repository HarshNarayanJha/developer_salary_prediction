# %% Cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% Cell 2
df = pd.read_csv("data/survey_results_public.csv")
print(df.head())

# %% Cell 3
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "CompTotal"]]
df = df.rename({"CompTotal": "Salary"}, axis=1)
# print(df.head())

# %% Cell 4
df = df[df["Salary"].notnull()]
# print(df.head())

# %% Cell 5
df.info()

# %% Cell 6
df = df.dropna()
df.isnull().sum()

# %% Cell 7
df = df[df["Employment"] == "Employed, full-time"]
df = df.drop("Employment", axis=1)

# %% Cell 8
df["Country"].value_counts()

# %% Cell 9


def shorten_categories(categories, cutoff):
    categorical_map = {}

    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map


# %% Cell 10

country_map = shorten_categories(df.Country.value_counts(), 200)
df["Country"] = df["Country"].map(country_map)
print(df.Country.value_counts())


# %% Cell 11

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot("Salary", "Country", ax=ax)
plt.suptitle("Salary (US$) v Country")
plt.title("")
plt.ylabel("Salary")
plt.ylim(0.0, 5000000)
plt.xticks(rotation=90)
plt.show()

# %% Cell 12
df = df[df["Salary"] <= 9000000]
df = df[df["Salary"] >= 50000]
df = df[df["Country"] != "Other"]
# %% Cell 13

df["YearsCodePro"] = df["YearsCodePro"].apply(
    lambda x: 50
    if x == "More than 50 years"
    else 0.5
    if x == "Less than 1 year"
    else float(x)
)
df["YearsCodePro"].unique()

# %% Cell 14


def clean_education(x):
    if "Bachelor" in x:
        return "Bachelor's degree"
    elif "Master" in x:
        return "Master's degree"
    elif "Professional" in x:
        return "Post grad"
    elif "Something else" or "without" in x:
        return "No degree"

    return "Elementry"


df["EdLevel"] = df["EdLevel"].apply(clean_education)
print(df["EdLevel"].value_counts())

# %% Cell 15

from sklearn.preprocessing import LabelEncoder

le_education = LabelEncoder()
df["EdLevel"] = le_education.fit_transform(df["EdLevel"])
df["EdLevel"].value_counts()
# %% Cell 16
le_country = LabelEncoder()
df["Country"] = le_country.fit_transform(df["Country"])
df["Country"].value_counts()

# %% Cell 17

X = df.drop("Salary", axis=1)
y = df["Salary"]

print(X.shape, y.shape)

# %% Cell 18

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, y_train)

# %% Cell 19

y_pred = reg.predict(X_test)

# %% Cell 20

from sklearn.metrics import mean_squared_error, mean_absolute_error

error = np.sqrt(mean_squared_error(y_test, y_pred))
print(error)

# %% Cell 21

from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

# %% Cell 22

y_pred = reg.predict(X_test)

# %% Cell 23
from sklearn.metrics import mean_squared_error, mean_absolute_error

error = np.sqrt(mean_squared_error(y_test, y_pred))
print(error)

# %% Cell 24

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# %% Cell 25

y_pred = reg.predict(X_test)

# %% Cell 26
from sklearn.metrics import mean_squared_error, mean_absolute_error

error = np.sqrt(mean_squared_error(y_test, y_pred))
print(error)

# %% Cel 27
from sklearn.model_selection import GridSearchCV

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}

reg = RandomForestRegressor()
gs = GridSearchCV(reg, parameters, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)

# %% Cell 28

reg = gs.best_estimator_

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_pred))
print(error)


# %% Cell 29

X_app = np.array([["India", "Bachelor's degree", 0]])
X_app
X_app[:, 0] = le_country.transform(X_app[:, 0])
X_app[:, 1] = le_education.transform(X_app[:, 1])
X_app = X_app.astype(float)
X_app
# %% Cell 30

y_app_pred = reg.predict(X_app)
y_app_pred

# %% Cell 31

import pickle

data = {
    'model': reg,
    'le_country': le_country,
    'le_education': le_education,
}

with open('model.pkl', 'wb') as file:
    pickle.dump(data, file)
