import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier, \
    VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("./titanic/train.csv")
test = pd.read_csv("./titanic/test.csv")

# generate browse with data
print(train.shape, "\n")
print(test.shape, "\n")
print(train.columns, "\n")
print(test.columns, "\n")
print(train.dtypes, "\n")
print(test.dtypes, "\n")

train.fillna(np.nan)
test.fillna(np.nan)
print(train.isnull().sum())
print(test.isnull().sum())

# value cleaning up and feature engineering
for ds in [train, test]:
    # some obvious features
    ds["Sex"] = ds["Sex"].map({"male": 0, "female": 1})
    ds["Age"] = ds["Age"].fillna(ds["Age"].median())
    ds["Fare"] = ds["Fare"].fillna(ds["Fare"].median())

    # maybe usefull ?
    ds["SibSp"] = ds["SibSp"].fillna(ds["SibSp"].median())
    ds["Parch"] = ds["Parch"].fillna(ds["Parch"].median())

# prepare data
features = ["Sex", "Age", "Fare", "SibSp", "Parch"]
Y = train["Survived"]
X = train.filter(features, axis=1)
Z = test.filter(features, axis=1)

print(X.sample(2), "\n")
print(Y.sample(2), "\n")
print(Z.sample(2), "\n")
print(X.shape, "\n")
print(Y.shape, "\n")
print(Z.shape, "\n")
print(X.isnull().sum())
print(Y.isnull().sum())
print(Z.isnull().sum())

# train
kfold = StratifiedKFold(n_splits=10)

classifiers = {}
classifiers['gbc'] = GradientBoostingClassifier()
classifiers['svc'] = SVC(probability=True)
classifiers['dtc'] = DecisionTreeClassifier()
classifiers['rfc'] = RandomForestClassifier()
classifiers['abc'] = AdaBoostClassifier()

best_classifiers = classifiers;

# benchmark
for name, clf in classifiers.items():
    res = cross_val_score(clf, X, Y, scoring="accuracy", cv=kfold)
    print(name, "\t", res.mean(), "\t", res.std())

# performance tuning, need to tune base classifiers

fitting_params = {}
fitting_params['svc'] = {'kernel': ['rbf', 'sigmoid', 'linear'],
                         'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                         'C': [1, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
                         }
fitting_params['gbc'] = {'loss': ["deviance"],
                         'n_estimators': [100, 150, 200, 250, 300],
                         'learning_rate': [0.5, 0.1, 0.05, 0.01, 0.005],
                         'max_depth': [4, 8, 16],
                         'min_samples_leaf': [100, 150, 200],
                         'max_features': [0.5, 0.3, 0.2, 0.1, 0.05]
                         }

for name, grid in fitting_params.items():
    search = GridSearchCV(classifiers[name],
                          param_grid=grid,
                          cv=kfold,
                          scoring="accuracy",
                          )

    search.fit(X, Y)
    best_classifiers[name] = search.best_estimator_
    print(name, "\n", search.best_params_, "\n", search.best_score_, "\n")

votingC = VotingClassifier(estimators=best_classifiers.items(), voting='soft')
votingC = votingC.fit(X, Y)
print(votingC.score(X, Y), "\n")

P = votingC.predict(Z)
StackingSubmission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': P})
StackingSubmission.to_csv("./titanic1.csv", index=False)
