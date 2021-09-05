



from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

df2=pd.read_csv("Biomechanical_Data_2Classes.csv")
# //USING BIOCHEMICAL DATA 2//
x2=df2.drop(['class'],axis=1)
y2=df2[['class']]

df3=df2.replace('Abnormal',1)
df3=df2.replace('Normal',0)

# //DROPPING THE COLUMN PRESENT AT TOP OF DECISION TREE//

# we can see that from the dataset 2 every decision tree top has the
# class of degree_spondylolisthesis for all min_samples_leaf=8, 16, 32

# //TRUNCATED DATASET//

x2=x2.drop(['degree_spondylolisthesis'],axis=1)

# //SPLITTING OF DATASET//

xtrain,xtest,ytrain,ytest=ms.train_test_split(x2,y2,train_size=0.77,test_size=0.22,random_state=50)


# //DECISION TREE1 WITH min_samples_leaf=8//

dt1 = DecisionTreeClassifier(random_state=0,min_samples_leaf=8)
dt1=dt1.fit(xtrain,ytrain)
ypred1=dt1.predict(xtest)

# //TRAINING LOSS//
ytrainpred1=dt1.predict(xtrain)
correcttrain1=np.sum(ytrain['class']!=ytrainpred1)/len(ytrain)

# //TESTING LOSS//
ytestpred1=dt1.predict(xtest)
correcttest1=(np.sum(ytest['class']!=ytestpred1))/len(ytest)


# //CALCULATING PROBABILITY//
yprob1=dt1.predict_proba(xtest)

# //CONSIDERING EACH CLASS//
yprob11=yprob1[:,0]
yprob12=yprob1[:,1]

# // PLOTTING DECISION TREE1//
fig1, ax1 = plt.subplots(figsize=(14,8))
tree.plot_tree(dt1, fontsize=6,feature_names=x2.columns,class_names=["Abnormal","Normal"])
plt.show()

nc1=dt1.tree_
print("No of nodes in Decision Tree with min_samples_leaf=8 is",nc1.node_count)


# //DECISION TREE2 WITH min_samples_leaf=16//
dt2 = DecisionTreeClassifier(random_state=0,min_samples_leaf=16)
dt2=dt2.fit(xtrain,ytrain)
ypred2=dt2.predict(xtest)


# //TRAINING LOSS//
ytrainpred2=dt2.predict(xtrain)
correcttrain2=np.sum(ytrain['class']!=ytrainpred2)/len(ytrain)

# //TESTING LOSS//
ytestpred2=dt2.predict(xtest)
correcttest2=(np.sum(ytest['class']!=ytestpred1))/len(ytest)



# //CALCULATING PROBABILITY//
yprob2=dt1.predict_proba(xtest)

# //CONSIDERING EACH CLASS//
yprob21=yprob2[:,0]
yprob22=yprob2[:,1]

# // PLOTTING DECISION TREE2//
fig2, ax2 = plt.subplots(figsize=(14,8))
tree.plot_tree(dt2, fontsize=6,feature_names=x2.columns,class_names=["Abnormal","Normal"])
plt.show()

nc2=dt2.tree_
print("No of nodes in Decision Tree with min_samples_leaf=16 is",nc2.node_count)


# //DECISION TREE3 WITH min_samples_leaf=32//
dt3 = DecisionTreeClassifier(random_state=0,min_samples_leaf=32)
dt3=dt3.fit(xtrain,ytrain)
ypred3=dt3.predict(xtest)

# //TRAINING LOSS//
ytrainpred3=dt3.predict(xtrain)
correcttrain3=(np.sum(ytrain['class']!=ytrainpred3))/len(ytrain)

# //TESTING LOSS//
ytestpred3=dt3.predict(xtest)
correcttest3=np.sum(ytest['class']!=ytestpred3)/len(ytest)



# //CALCULATING PROBABILITY//
yprob3=dt1.predict_proba(xtest)

# //CONSIDERING EACH CLASS//
yprob31=yprob3[:,0]
yprob32=yprob3[:,1]

nc3=dt3.tree_
print("No of nodes in Decision Tree with min_samples_leaf=32 is",nc3.node_count)

# //Training Loss v/s Testing Loss Plot//

plt.figure(figsize=(10,5))
plt.bar(x=['8node_train','8node_test','-','16node_train','16node_test','*','32node_train','32node_test'],height=[correcttrain1,correcttest1,0,correcttrain2,correcttest2,0,correcttrain3,correcttest3])
plt.title("Training Loss v/s Testing Loss for all Decision trees")

plt.ylabel("Misclassfication rate")
plt.show()


# // PLOTTING DECISION TREE3//
fig3, ax3 = plt.subplots(figsize=(14,8))
tree.plot_tree(dt3, fontsize=6,feature_names=x2.columns,class_names=["Abnormal","Normal"])
plt.show()

# ANS 3 B.

# //CONFUSION MATRIX//

# //CONFUSION MATRIX FOR DECISION TREE1//
cm1=confusion_matrix(ytest,ypred1)
print("Confusion Matrix for decision tree1",cm1)

# //CONFUSION MATRIX FOR DECISION TREE2//
cm2=confusion_matrix(ytest,ypred2)
print("Confusion Matrix for decision tree2",cm2)

# //CONFUSION MATRIX FOR DECISION TREE3//
cm3=confusion_matrix(ytest,ypred3)
print("Confusion Matrix for decision tree1",cm3)


print('Precision, Recall for Decision Tree1 is')
print(metrics.classification_report(ytest, ypred1, digits=3))

print('Precision, Recall for Decision Tree2 is')
print(metrics.classification_report(ytest, ypred2, digits=3))


print('Precision, Recall for Decision Tree3 is')
print(metrics.classification_report(ytest, ypred3, digits=3))


# //ACCURACY FOR DECISION TREE1//
a1=accuracy_score(ytest,ypred1,normalize=False)
print("Accuracy for decision tree1",a1)

# //ACCURACY FOR DECISION TREE2//
a2=accuracy_score(ytest,ypred2,normalize=False)
print("Accuracy for decision tree1",a2)

# //ACCURACY FOR DECISION TREE3//
a3=accuracy_score(ytest,ypred3,normalize=False)
print("Accuracy for decision tree1",a3)


# // PRECISION RECALL CURVE FOR Abnormal CLASS  OF DECISION TREE1//
ytrue=np.array(ytest['class'],dtype='object')
ytrue=np.where(ytrue=="Abnormal",1,0)
precision11, recall11, thresholds11 = precision_recall_curve(ytrue, yprob11,pos_label=None)

# // PRECISION RECALL CURVE FOR Normal CLASS  OF DECISION TREE1//
ytrue=np.array(ytest['class'],dtype='object')
ytrue=np.where(ytrue=="Normal",1,0)

precision12, recall12, thresholds12 = precision_recall_curve(ytrue, yprob12,pos_label=None)

#
# # // PRECISION RECALL CURVE FOR Abnormal CLASS OF DECISION TREE2//
ytrue=np.array(ytest['class'],dtype='object')
ytrue=np.where(ytrue=="Abnormal",1,0)
precision21, recall21, thresholds21 = precision_recall_curve(ytrue, yprob21,pos_label=None)

#
# # // PRECISION RECALL CURVE FOR Normal CLASS  OF DECISION TREE2//
ytrue=np.array(ytest['class'],dtype='object')
ytrue=np.where(ytrue=="Normal",1,0)
precision22, recall22, thresholds22 = precision_recall_curve(ytrue, yprob22,pos_label=None)

#
#
# # // PRECISION RECALL CURVE FOR Abnormal CLASS  OF DECISION TREE3//
ytrue=np.array(ytest['class'],dtype='object')
ytrue=np.where(ytrue=="Abnormal",1,0)
precision31, recall31, thresholds31 = precision_recall_curve(ytrue, yprob31,pos_label=None)

#
# # // PRECISION RECALL CURVE FOR Normal CLASS  OF DECISION TREE3//
ytrue=np.array(ytest['class'],dtype='object')
ytrue=np.where(ytrue=="Normal",1,0)
precision32, recall32, thresholds32 = precision_recall_curve(ytrue, yprob32,pos_label=None)



# ///Adding small noise factor to distinguish the superimposing curves from each other///

l=[recall11, precision11,recall12, precision12,recall21, precision21,recall22, precision22,recall31, precision31,recall32, precision32,]
recall11+=np.random.uniform(0.03,0.07,len(recall11))
precision11+=np.random.uniform(0.03,0.07,len(precision11))
recall12+=np.random.uniform(0.03,0.07,len(recall12))
precision12+=np.random.uniform(0.03,0.07,len(precision12))
recall21+=np.random.uniform(0.03,0.07,len(recall21))
precision21+=np.random.uniform(0.03,0.07,len(precision21))
recall22+=np.random.uniform(0.03,0.07,len(recall22))
precision22+=np.random.uniform(0.03,0.07,len(precision22))




plt.figure(figsize=(30,35))

plt.plot(recall11, precision11, lw=2, color='red')
plt.plot(recall12, precision12, lw=2, color='skyblue')
plt.plot(recall21, precision21, lw=2, color='olive')
plt.plot(recall22, precision22, lw=2, color='yellow')

plt.plot(recall31, precision31, lw=2, color='green')
plt.plot(recall32, precision32, lw=2, color='black')

plt.legend(['Precision Recall curve for 8 node decision tree for Abnormal class',
            'Precision Recall curve  for 8 node decision tree for Normal class',
            'Precision Recall curve  for 16 node decision tree for Abnormal class',
            'Precision Recall curve  for 16 node decision tree for Normal class',
            'Precision Recall curve  for 32 node decision tree for Abnormal class',
            'Precision Recall curve  for 32 node decision tree for Normal class'
            ])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PRECISION RECALL CURVE')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
plt.grid()
plt.show()

# #