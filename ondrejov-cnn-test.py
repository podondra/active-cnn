from active_cnn import data
from active_cnn.model import CNN
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


ids, X, labels, y = data.get_ondrejov_dataset('data/ondrejov-dataset.csv')
# preprocesing
minmax_scale(X, feature_range=(-1, 1), axis=1, copy=False)

# create train and test set
splitting = train_test_split(X, y, stratify=y)
X_train, X_test, y_train, y_test = splitting
    
cnn = CNN()
cnn.train(X_train, y_train)
y_pred = cnn.predict(X_test)

print('confusion matrix')
print(confusion_matrix(y_test, y_pred > 0.5))
