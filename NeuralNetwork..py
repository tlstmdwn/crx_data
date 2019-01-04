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



import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
len(X_train.columns)
def he_init(n_inputs):
    stddev = tf.sqrt(2 / n_inputs)
    return keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev, seed=None)


model = keras.Sequential()
model.add(keras.layers.Dense(units=20, activation='relu', kernel_initializer = he_init(46)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=20, activation='relu',  kernel_initializer = he_init(20)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=20, activation='relu', kernel_initializer = he_init(20)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=20, activation='relu', kernel_initializer = he_init(20)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=2, activation=tf.nn.softmax,  kernel_initializer = he_init(20)))

model.compile(optimizer=tf.train.AdamOptimizer(),  # adam learning rate 업데이트
              loss='sparse_categorical_crossentropy', # loss function 결정
              metrics=['accuracy'])  # 결정 metric 결정
 model.fit(np.array(X_train), np.array(y_train), epochs=300, batch_size = 50)

#metrics
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
print(test_loss, test_acc)
y_pred = np.argmax(model.predict(np.array(X_test)),axis=1)
y_pred_proba = model.predict(np.array(X_test))
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, 'o-', label="XGB")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC curve of XGB')
plt.show()
auc(fpr, tpr)
              
             
             
