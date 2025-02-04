import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
data.drop(['Id'], axis=1, inplace= True)
df_test.drop(['Id'], axis=1, inplace= True)

# remove feature has more than 50% missing value
removed_feature = []
for feature in data.columns:
    missing_value = data[feature].isnull().sum()
    missing_percentages = missing_value/len(data)*100
    if missing_percentages >= 50:
        removed_feature.append(feature)
        data.drop([feature], axis = 1, inplace=True)
        df_test.drop([feature], axis = 1, inplace=True)

# dữ liệu kiểu thời gian
temporal_feature = [feature for feature in data.columns if 'Yr' in feature or 'Year' in feature]
# numerical featurefeature
numerical_feature = data.select_dtypes(include=['number']).columns.tolist()
numerical_feature = [feature for feature in numerical_feature if feature not in ['SalePrice'] + temporal_feature]
# categorical featurefeature
categorical_feature = data.select_dtypes(include=['object', 'category']).columns.tolist()




# check đa cộng tuyếntuyến   
correlation_matrix = data[numerical_feature].corr()
threshold = 0.8
high_corr = correlation_matrix.stack().reset_index()
high_corr = high_corr[(high_corr[0] > threshold) & (high_corr['level_0'] != high_corr['level_1'])]

# Hiển thị các cặp feature có tương quan cao
print("Cặp features có tương quan cao:")
# print(high_corr)
# loại bỏ đa cộng tuyếntuyến
data.drop(['1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], axis = 1, inplace=True)
df_test.drop(['1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], axis = 1, inplace=True)


# refine numerical_feature
numerical_feature = [feature for feature in numerical_feature if feature 
not in ['1stFlrSF', 'TotRmsAbvGrd', 'GarageArea']]


# split data
x = data.drop(['SalePrice'], axis = 1)
y = data['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)




num_transform = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy= 'median')),
    ('scaler', StandardScaler())
])

temporal_transform = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy= 'median')),
    ('scaler', MinMaxScaler())
])

one_transform = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transform, numerical_feature),
    ('temporal', temporal_transform, temporal_feature),
    ('one', one_transform, categorical_feature)
])


model = Pipeline(steps= [
    ('preprocessor', preprocessor),
    ('linear', LinearRegression())
])


model.fit(x_train, y_train)
y_pred = model.predict(x_test)


print(f'r2_score: {r2_score(y_test, y_pred)}')
print(f'mean_squared_error: {mean_squared_error(y_test, y_pred)}')

# create the DataFrame

data_output = {"True Value" : y_test, "Predicted Value" : y_pred}
df = pd.DataFrame(data_output)

# Display the DataFrame
print(df)











