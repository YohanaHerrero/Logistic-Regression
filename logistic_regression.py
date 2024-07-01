import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss

#open dataset
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

#Let's select some features for the modeling. Also, change the data type of churn to be an integer (sklearn requirement)
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

#define X and y
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

#normalize dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

#split dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

#create the model
#solver: different numerical optimizers to find parameters; C: Regularization is a technique used to solve the overfitting 
#problem. C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify 
#stronger regularization.
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

#Make predictions
y_predict = LR.predict(X_test)

#estimate the probability for each class, ordered by the label of classes. So, the first column is the probability of class 0, 
#P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
y_predict_prob = LR.predict_proba(X_test)

#evaluate model accuracy
#with the jaccard method = size of the intersection divided by the size of the union of the two label sets. If the entire set 
#of predicted labels strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0
jaccard_score(y_test, y_predict, pos_label=0)

#another way to evaluate the accuracy is with the confussion matrix
#function to plot the confussion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, y_predict, labels=[1,0]))

#compute confussion matrix
cnf_matrix = confusion_matrix(y_test, y_predict, labels=[1,0])
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'], normalize= False,  title='Confusion matrix')
print (classification_report(y_test, y_predict))

#Another way to evaluate the accuracy is with the log loss (Logarithmic loss)
#In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). This probability is a 
#value between 0 and 1. Log loss measures the performance of a classifier where the predicted output is a probability value 
#between 0 and 1.
print ("LogLoss: : %.2f" % log_loss(y_test, y_predict_prob))

#we could now use different solvers for the logistic regression model and see if we improve the accuracy of the model
#eg using 'sag' instead of 'liblinear'
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
y_predict_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, y_predict_prob2))
