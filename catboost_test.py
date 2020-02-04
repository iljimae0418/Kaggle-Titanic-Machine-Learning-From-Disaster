import numpy as np
import pandas as pd
import hyperopt
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read in data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
# fill missing values for Catboost to recognize
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)
# get labels for train data
x = train.drop('Survived',axis=1)
y = train.Survived
# choose features to be trained
features_idx = np.where(x.dtypes != float)[0]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.85,random_state=1234)
# create catboost model
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
model.fit(xtrain,ytrain,cat_features=features_idx,eval_set=(xtest,ytest))
'''
# 10-fold cv
cv_data = cv(Pool(x,y,cat_features=features_idx),model.get_params(),fold_count=10)
# print accuracy for the cv and model
print('best cv accuracy : {}'.format(np.max(cv_data["b'Accuracy'_test_avg"])))
print('the model test accuracy is :{}'.format(accuracy_score(ytest,model.predict(xtest))))
'''
# make submission
pred = model.predict(test)
pred = pred.astype(np.int)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred})
submission.to_csv('titanic_catboost.csv',index=False)
