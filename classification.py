import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from lazypredict.Supervised import LazyClassifier, LazyRegressor
data = pd.read_csv('diabetes.csv')
target = 'Outcome'

# profile = ProfileReport(data, title="Report", explorative=True)
# profile.to_file('data_report.html')
x = data.drop(target, axis = 1)
y = data[target]

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42)

#Preprocess data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

params = {
    'n_estimators': [50, 75, 100, 200],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5, 10]
}
#Pick model
model = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=params,
    n_iter=20,
    scoring='recall',
    cv=6,
    verbose=1,
    n_jobs=1
)
# model = GridSearchCV(
#     RandomForestClassifier(random_state=42),
#     param_grid=params,
#     scoring='recall',
#     cv=6,
#     verbose=1,
#     n_jobs=1
# )
# model = SVC(random_state= 42)
# model = RandomForestClassifier(random_state= 42)
# model = LogisticRegression(random_state= 42)
#Train model
model.fit(x_train, y_train)
print(model.best_score_)
print(model.best_params_)

y_predict = model.predict(x_test)
# for i, j in zip(y_predict, y_test.values):
# print("Predicted value: {}. Actual value: {}".format(i, j))
# print("Accuracy: {}".format(accuracy_score(y_test, y_predict)))
# print("Precision: {}".format(precision_score(y_test, y_predict)))
# print("Recall: {}".format(recall_score(y_test, y_predict)))
# print("F1 Score: {}".format(f1_score(y_test, y_predict)))
# print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
# print(y_predict)


# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
