import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier, \
    VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
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

# value cleaning up
for ds in [train, test]:
    ds["Sex"] = ds["Sex"].map({"male": 0, "female": 1})
    ds["Age"] = ds["Age"].fillna(ds["Age"].median())
    ds["Fare"] = ds["Fare"].fillna(ds["Fare"].median())

# prepare data
Y = train["Survived"]
X = train.filter(["Sex", "Age", "Fare"], axis=1)
Z = test.filter(["Sex", "Age", "Fare"], axis=1)

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
classifiers['nnc'] = MLPClassifier()
classifiers['rfc'] = RandomForestClassifier()
classifiers['abc'] = AdaBoostClassifier()

for name, clf in classifiers.items():
    res = cross_val_score(clf, X, Y, scoring="accuracy", cv=kfold)
    print(name, "\t", res.mean(), "\t", res.std())

votingC = VotingClassifier(estimators=classifiers.items(), voting='soft')
votingC = votingC.fit(X, Y)
print(votingC.score(X, Y), "\n")

P = votingC.predict(Z)
StackingSubmission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': P})
StackingSubmission.to_csv("./titanic1.csv", index=False)
