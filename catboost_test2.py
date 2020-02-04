import numpy as np
import pandas as pd
import hyperopt
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read in data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
# add features
train['Boy'], test['Boy'] = [(df.Name.str.split().str[1] == 'Master.').astype('int') for df in [train, test]]
train['Surname'], test['Surname'] = [df.Name.str.split(',').str[0] for df in [train, test]]
# fit model
model = CatBoostClassifier(one_hot_max_size=4, iterations=100, random_seed=0, verbose=False)
model.fit(train[['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']].fillna(''), train['Survived'], cat_features=[0, 2, 4])
# make prediction
pred = model.predict(test[['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']].fillna('')).astype('int')
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred})
submission.to_csv('titanic_catboost_bulk.csv',index=False)
