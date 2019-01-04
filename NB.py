#Multinomial, Gaussian 혼용

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


#category변수랑 numerical 나누기


from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
GauNB = model.fit(X_train_nume, y_train)
Gaupred = GauNB.predict_proba(X_test_nume)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
MulNB = model.fit(X_train_cate, y_train)
Mulpred = MulNB.predict_proba(X_test_cate)
Allpred = Gaupred*Mulpred
y_pred = np.argmax(Allpred, axis = 1)
print(np.sum(y_pred == y_test)/len(y_test))
print(f1_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, 'o-', label="XGB")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of XGB')
plt.show()
auc(fpr, tpr)
