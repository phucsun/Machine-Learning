import pandas as pd
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('StudentScore.xls')
# profile = ProfileReport(data, title='Student Score Report', explorative=True)
# profile.to_file("student_score_report.html")

target = 'math score'
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
])

education_levels = ['some high school', 'high school', 'some college',
                    "associate's degree", "bachelor's degree", "master's degree"]
genders = x_train["gender"].unique()
lunchs = x_train["lunch"].unique()
test_preps = x_train["test preparation course"].unique()


ord_tranformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("endcoder", OrdinalEncoder(categories=[education_levels, genders, lunchs, test_preps]))
])

nom_tranformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num_features", num_transformer, ['reading score', 'writing score']),
        ("ord_bool_features", ord_tranformer, ['parental level of education', 'gender', 'lunch', 'test preparation course']),
        ("nom_features", nom_tranformer, ['race/ethnicity'])
    ]
)

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("radom_forest", RandomForestRegressor(random_state= 42))
])

params = {
    'radom_forest__n_estimators': [50, 100, 200],
    'radom_forest__criterion': ['absolute_error', 'poisson', 'friedman_mse', 'squared_error'],
}
model = GridSearchCV(
    reg,
    param_grid=params,
    scoring= 'r2',
    cv=6,
    verbose=2,
    n_jobs=-1
)


model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
# print(model.best_score_)
# print(model.best_params_)
# print("MAE:{}".format(mean_absolute_error(y_test, y_predict)))
# print("MSE:{}".format(mean_squared_error(y_test, y_predict)))
# print("R2:{}".format(r2_score(y_test, y_predict)))

fake_data = pd.DataFrame.from_dict({
    "gender": ["male"],
    "race/ethnicity": ["group A"],
    "parental level of education": ["master's degree"],
    "lunch": ["standard"],
    "test preparation course": ["none"],
    "reading score": [100],
    "writing score": [100]
})
print(model.predict(fake_data))

