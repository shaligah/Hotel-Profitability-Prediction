#IMPORTING RELEVANT LIBRARIES

#data manipulation
import pandas as pd
import numpy as np

#data preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#model building
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

#model evaluation
from sklearn import metrics

train_df = pd.read_csv('CE802_P2_Data.csv')
train_df.head()

#checking data types
train_df.dtypes.value_counts()

#checking for null cells
train_df.isnull().mean()*100

msno.matrix(train_df)


# since 50% of F21 is missing we drop the column entirely because filling it with an appropriate value might distort the results of the study

train_df.drop(columns='F21',inplace=True)

enc =LabelEncoder()
train_df['Class']= enc.fit_transform(train_df['Class'])

X,y = train_df.loc[:,train_df.columns!='Class'], train_df.loc[:,'Class']

#SPLITTING THE DATA INTO TRAINING AND TESTING for model building and evaluation
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=0)
X_test.shape, y_test.shape


# # MODEL BUILDING WITH ALL FEATURES

# the model is built using four distinct classifiers to comparison and better performance

#Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
print("Model Accuracy:", metrics.accuracy_score(y_test, y_prediction) )

#scaling data for appropriate models
scaler = MinMaxScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.fit_transform(X_test)

#LLOGISTIC REGRESSION CLASSIFIER
clf2 = LogisticRegression()
clf2.fit(X_train1,y_train)
y2prediction = clf2.predict(X_test1)
confu_matrix = metrics.confusion_matrix(y_test,y2prediction)
print("Model Accuracy:", metrics.accuracy_score(y_test, y2prediction) )

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

# create heatmap
sns.heatmap(pd.DataFrame(confu_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True')
plt.xlabel('Predicted')


#SVM CLASSIFICATION
#TUNING THE HYPERPARAMETERS
C = [0.1,1, 10, 100]
gamma = [10, 1, 0.1, 0.01, 0.001, 0.0001]
kernel = [ 'sigmoid', 'poly', 'rbf']
p_grid = dict(C=C, gamma=gamma, kernel=kernel)
n_grid = GridSearchCV(SVC(), param_grid=p_grid, refit=True, verbose=2)
n_grid.fit(X_train1,y_train)
print(n_grid.best_params_)

clf3 =  SVC(C=1, gamma=1, kernel='poly')
clf3.fit(X_train1, y_train)
y4predict = clf3.predict(X_test1)
results = metrics.classification_report(y_test,y4predict)
accuracy = metrics.accuracy_score(y_test, y4predict)
print(accuracy)


#XGBOOST Classifier
model1 = xgb.XGBClassifier()
model1.fit(X_train, y_train)
p = model1.predict(X_test)
metrics.accuracy_score(y_test, p)


# # comparing the Accuracy of the 3 classifiers
Y_values = [metrics.accuracy_score(y_test, y_prediction), metrics.accuracy_score(y_test, y2prediction), metrics.accuracy_score(y_test, y4predict), metrics.accuracy_score(y_test, p)]
X_values = ['Decision Tree', 'Logistic', 'SVM', 'XGBoost']
plt.bar(X_values, Y_values, color = 'maroon', width=0.4)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores')
plt.show()


# # MODEL BUILDING WITH FEATURE SELECTION

#Checking for multicollinearity
train_df.corr()

plt.figure(figsize=(20,10))
sns.heatmap(train_df.corr(),annot=True)


# from the matrix we can see that some variables share a high correlation with other independent variables suggesting multicollinearity as such we investigate further with the variance inflation factor(VIF)

vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                   for i in range(len(X_train.columns))]

vif_data


# the VIF confirms there is high correlation between one factor with other factors as such we tackle this problem by singularly removing variables with high VIF values.

Xt2 = X_train.copy()
Xt2.drop(columns=['F4','F13','F15','F16','F10','F19','F2','F5','F14','F6'], inplace=True)
vif_data2 = pd.DataFrame()
vif_data2["feature"] = Xt2.columns
vif_data2["VIF"] = [variance_inflation_factor(Xt2.values, i) 
                   for i in range(len(Xt2.columns))]


vif_data2

X_test2 = X_test.loc[:,Xt2.columns]
X_test2.shape,Xt2.shape


#Decision Tree Classifier
clf4 = tree.DecisionTreeClassifier()
clf4.fit(Xt2,y_train)
pred5 = clf4.predict(X_test2)
metrics.accuracy_score(y_test,pred5)


#XGBoost Classifier
model2 = xgb.XGBClassifier()
model2.fit(Xt2,y_train)
pred6 = model2.predict(X_test2)
metrics.accuracy_score(y_test,pred6)

X_train2 = scaler.fit_transform(Xt2)
X_test3 = scaler.fit_transform(X_test2)


#SVM Classifier
clf5 =  SVC(C=1, gamma=1, kernel='poly')
clf5.fit(X_train2,y_train)
pred7 = clf5.predict(X_test3)
results2 = metrics.classification_report(y_test,pred7)
accuracy2 = metrics.accuracy_score(y_test, pred7)
print(accuracy2)


#Logistic Regression Classifier
clf6 = LogisticRegression()
clf6.fit(X_train2,y_train)
pred7 = clf6.predict(X_test3)
metrics.accuracy_score(y_test,pred7)


# We can conclude that the XGBoost Classifier performed best amongst all the models and even better with feature selection as such when predicting for this particular scenario we implement XGboost with selected features
