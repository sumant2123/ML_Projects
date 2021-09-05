



from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn import metrics


df3=pd.read_csv("Biomechanical_Data_3Classes.csv")

#//USING BIOCHEMICAL DATASET 3//
#
x3=df3.drop(['class'],axis=1)
y3=df3[['class']]

# //SPLITTING DATASET//

xtrain1,xtest1,ytrain1,ytest1=ms.train_test_split(x3,y3,train_size=0.77,test_size=0.22,random_state=50)

# //DECISION TREE1 WITH min_samples_leaf=8//
dt11 = DecisionTreeClassifier(random_state=0,min_samples_leaf=8)
dt11=dt11.fit(xtrain1,ytrain1)
ypred11=dt11.predict(xtest1)

# //TRAINING LOSS//
ytrainpred11=dt11.predict(xtrain1)
correcttrain11=np.sum(ytrain1['class']!=ytrainpred11)/len(ytrain1)

# //TESTING LOSS//
ytestpred11=dt11.predict(xtest1)
correcttest11=(np.sum(ytest1['class']!=ytestpred11))/len(ytest1)



# //CALCULATING PROBABILITY//
yprob11=dt11.predict_proba(xtest1)

# //CONSIDERING EACH CLASSS//
yprob111=yprob11[:,0]
yprob112=yprob11[:,1]
yprob113=yprob11[:,2]

# //PLOT DECISION TREE1//
fig11, ax11 = plt.subplots(figsize=(14,8))
tree.plot_tree(dt11, fontsize=6,feature_names=x3.columns,class_names=["Hernia","Spondylolisthesis","Normal"])
plt.show()

# //DECISION TREE2 WITH min_samples_leaf=16//
dt12 = DecisionTreeClassifier(random_state=0,min_samples_leaf=16)
dt12=dt12.fit(xtrain1,ytrain1)
ypred12=dt12.predict(xtest1)

# //TRAINING LOSS//
ytrainpred12=dt12.predict(xtrain1)
correcttrain12=np.sum(ytrain1['class']!=ytrainpred12)/len(ytrain1)

# //TESTING LOSS//
ytestpred12=dt12.predict(xtest1)
correcttest12=(np.sum(ytest1['class']!=ytestpred12))/len(ytest1)



# //CALCULATING PROBABILITY//
yprob21=dt12.predict_proba(xtest1)


# //CONSIDERING EACH CLASSS//
yprob211=yprob21[:,0]
yprob212=yprob21[:,1]
yprob213=yprob21[:,2]


# //PLOT DECISION TREE2//
fig12, ax12 = plt.subplots(figsize=(14,8))
tree.plot_tree(dt12, fontsize=6,feature_names=x3.columns,class_names=["Hernia","Spondylolisthesis","Normal"])
plt.show()

#//DECISION TREE3 WITH min_samples_leaf=32//
dt13 = DecisionTreeClassifier(random_state=0,min_samples_leaf=32)
dt13=dt13.fit(xtrain1,ytrain1)
ypred13=dt13.predict(xtest1)

# //TRAINING LOSS//
ytrainpred13=dt13.predict(xtrain1)
correcttrain13=np.sum(ytrain1['class']!=ytrainpred13)/len(ytrain1)

# //TESTING LOSS//
ytestpred13=dt12.predict(xtest1)
correcttest13=(np.sum(ytest1['class']!=ytestpred13))/len(ytest1)



# //CALCULATING PROBABILITY//
yprob31=dt13.predict_proba(xtest1)


# //CONSIDERING EACH CLASSS//
yprob311=yprob31[:,0]
yprob312=yprob31[:,1]
yprob313=yprob31[:,2]

#//PLOT DECISION TREE3//
fig13, ax13 = plt.subplots(figsize=(14,8))
tree.plot_tree(dt13, fontsize=6,feature_names=x3.columns,class_names=["Hernia","Spondylolisthesis","Normal"])
plt.show()

# ANS 2 B.


# //CONFUSION MATRIX FOR DECISION TREE1//
cm11=confusion_matrix(ytest1,ypred11)
print("Confusion Matrix for decision tree1",cm11)

# //CONFUSION MATRIX FOR DECISION TREE2//
cm12=confusion_matrix(ytest1,ypred12)
print("Confusion Matrix for decision tree2",cm12)

# //CONFUSION MATRIX FOR DECISION TREE3//
cm13=confusion_matrix(ytest1,ypred13)
print("Confusion Matrix for decision tree1",cm13)

print('Precision, Recall for Decision Tree1 is')
print(metrics.classification_report(ytest1, ypred11, digits=3))

print('Precision, Recall for Decision Tree2 is')
print(metrics.classification_report(ytest1, ypred12, digits=3))


print('Precision, Recall for Decision Tree3 is')
print(metrics.classification_report(ytest1, ypred13, digits=3))

# //ACCURACY FOR DECISION TREE1//
a11=accuracy_score(ytest1,ypred11,normalize=False)
print("Accuracy for decision tree1",a11)

# //ACCURACY FOR DECISION TREE2//
a12=accuracy_score(ytest1,ypred12,normalize=False)
print("Accuracy for decision tree1",a12)

# //ACCURACY FOR DECISION TREE3//
a13=accuracy_score(ytest1,ypred13,normalize=False)
print("Accuracy for decision tree1",a13)



# // PRECISION RECALL CURVE FOR HERNIA CLASS OF DECISION TREE1//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Hernia",1,0)


precision, recall, thresholds = precision_recall_curve(ytrue, yprob111,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR HERNIA CLASS OF DECISION TREE1')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()
#
#
# // PRECISION RECALL CURVE FOR NORMAL CLASS OF DECISION TREE1//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Normal",1,0)



precision, recall, thresholds = precision_recall_curve(ytrue, yprob112,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR NORMAL CLASS OF DECISION TREE1')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()

#
#
# // PRECISION RECALL CURVE FOR Spondylolisthesis CLASS OF DECISION TREE1//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Spondylolisthesis",1,0)


precision, recall, thresholds = precision_recall_curve(ytrue, yprob113,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR Spondylolisthesis CLASS -DECISION TREE1')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()

#
# // PRECISION RECALL CURVE FOR HERNIA CLASS OF DECISION TREE2//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Hernia",1,0)


precision, recall, thresholds = precision_recall_curve(ytrue, yprob211,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR HERNIA CLASS OF DECISION TREE2')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()
#
#
# // PRECISION RECALL CURVE FOR NORMAL CLASS OF DECISION TREE2//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Normal",1,0)


precision, recall, thresholds = precision_recall_curve(ytrue, yprob212,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR NORMAL CLASS OF DECISION TREE2')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()

#

# // PRECISION RECALL CURVE FOR Spondylolisthesis CLASS OF DECISION TREE2//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Spondylolisthesis",1,0)


precision, recall, thresholds = precision_recall_curve(ytrue, yprob213,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR Spondylolisthesis CLASS -DECISION TREE2')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()



# // PRECISION RECALL CURVE FOR HERNIA CLASS OF DECISION TREE3//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Hernia",1,0)


precision, recall, thresholds = precision_recall_curve(ytrue, yprob311,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR HERNIA CLASS OF DECISION TREE3')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()
#
#
# // PRECISION RECALL CURVE FOR NORMAL CLASS OF DECISION TREE3//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Normal",1,0)

precision, recall, thresholds = precision_recall_curve(ytrue, yprob312,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR NORMAL CLASS OF DECISION TREE3')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()

#

# // PRECISION RECALL CURVE FOR Spondylolisthesis CLASS OF DECISION TREE3//
ytrue=np.array(ytest1['class'],dtype='object')
ytrue=np.where(ytrue=="Spondylolisthesis",1,0)

precision, recall, thresholds = precision_recall_curve(ytrue, yprob313,pos_label=None)
plt.plot(recall, precision, lw=2, color='navy')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE FOR Spondylolisthesis CLASS -DECISION TREE3')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()
