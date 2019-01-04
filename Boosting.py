import xgboost as xgb
from matplotlib import pyplot
cols =  ['A{}'.format(i) for i in range(1,17) if data['A{}'.format(i)].dtype == 'O']
for col in cols:
    data[col] = data[col].astype('category')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:15], data.A16, test_size=0.2)

X_train_nume = X_train[['A2', 'A3', 'A8', 'A11','A14', 'A15']]
X_train_cate = X_train[['A1', 'A4', 'A5', 'A7','A6', 'A9', 'A10', 'A12', 'A13']]
X_test_nume = X_test[['A2', 'A3', 'A8', 'A11','A14', 'A15']]
X_test_cate = X_test[['A1', 'A4', 'A5', 'A7','A6', 'A9', 'A10', 'A12', 'A13']]

cols= X_train_cate.columns
for col in cols:
    one_hot = pd.get_dummies(X_train_cate[col], prefix= col)
    X_train_cate = X_train_cate.join(one_hot)
    X_train_cate = X_train_cate.drop(col, axis=1)
    cols_test= X_test_cate.columns
for col in cols_test:
    one_hot = pd.get_dummies(X_test_cate[col], prefix= col)
    X_test_cate = X_test_cate.join(one_hot)
    X_test_cate = X_test_cate.drop(col, axis=1)
    
    
y_test = y_test.replace('-',1)
y_test = y_test.replace('+',0)
y_train = y_train.replace('-',1)
y_train = y_train.replace('+',0)
y_test = y_test.astype('category')
y_train = y_train.astype('category')
X_train = pd.merge(X_train_nume, X_train_cate, right_index = True, left_index=True)
X_test = pd.merge(X_test_nume, X_test_cate, right_index = True, left_index=True)

params = {'learning_rate': 0.1, 
          'max_depth':4, 
          'n_estimator':200,
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          
          'alpha':3,
          'objective': 'binary:logistic', 
          'random_state': 99, 
          
          'silent': True}
    
clf = xgb.XGBClassifier(**params)
clf.fit(X_train, y_train, eval_set = [(X_train,y_train)],
        eval_metric= 'error', verbose =True, early_stopping_rounds=30
         )
#중요도 확인
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()



#Metric확인
y_pred = clf.predict(X_test)
print(np.sum(y_pred == y_test)/len(y_test))
print(f1_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, 'o-', label="XGB")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of XGB')
plt.show()
auc(fpr, tpr)

