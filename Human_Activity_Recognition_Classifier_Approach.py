'''
Human Activation Recognition implementation example using sklearn library
sklearn tutorial: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html,
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html,
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
'''






import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
#load the data----------------------------------------------------------------------------------------------------------
test=pd.read_csv("test.csv")
train=pd.read_csv("train.csv")
#take the features and drop your activity-------------------------------------------------------------------------------
features = train.iloc[:,0:562]
label = train['Activity']

#feature selection procedures-------------------------------------------------------------------------------------------
clf = ExtraTreesClassifier()
clf = clf.fit(features, label)
model = SelectFromModel(clf, prefit=True)
features_new_extraclassifier = model.transform(features)

#L1-based feature selection---------------------------------------------------------------------------------------------
lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(features, label)
model_2 = SelectFromModel(lsvc, prefit=True)
features_linearSVC = model_2.transform(features)
#-----------------------------------------------------------------------------------------------------------------------
Classifiers = [DecisionTreeClassifier(),RandomForestClassifier(n_estimators=200),svm.SVC(C=2.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=8, shrinking=True,
    tol=0.001, verbose=False),linear_model.SGDClassifier()]
test_features= test.iloc[:,0:562]
Time_1=[]
Model_1=[]
Out_Accuracy_1=[]
for clf in Classifiers:
    print("Started")
    fit=clf.fit(features,label)
    pred=fit.predict(test_features)
    Model_1.append(clf.__class__.__name__)
    Out_Accuracy_1.append(accuracy_score(test['Activity'],pred))

print("Without feature selection:")
print("Model:",Model_1)
print("Out_Accuracy:",Out_Accuracy_1)


#Tree Based Classifier--------------------------------------------------------------------------------------------------
test_features= model.transform(test.iloc[:,0:562])
Time_2=[]
Model_2=[]
Out_Accuracy_2=[]
for clf in Classifiers:
    fit=clf.fit(features_new_extraclassifier,label)
    pred=fit.predict(test_features)
    Model_2.append(clf.__class__.__name__)
    Out_Accuracy_2.append(accuracy_score(test['Activity'],pred))

print("With Tree based feature selection")
print("Model:",Model_2)
print("Out_Accuracy:",Out_Accuracy_2)

# L1 Based classifier----------------------------------------------------------------------------------------------------
test_features= model_2.transform(test.iloc[:,0:562])
Time_3=[]
Model_3=[]
Out_Accuracy_3=[]
for clf in Classifiers:
    fit=clf.fit(features_linearSVC,label)
    pred=fit.predict(test_features)
    Model_3.append(clf.__class__.__name__)
    Out_Accuracy_3.append(accuracy_score(test['Activity'],pred))
# -----------------------------------------------------------------------------------------------------------------------
print("With L1 based feature selection")
print("Model:",Model_3)
print("Out_Accuracy:",Out_Accuracy_3)
