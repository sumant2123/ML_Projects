import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.linear_model import RidgeCV,LassoCV
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold

# Reading CSV file and converting it into dataframe
data=pd.read_csv('hm2data.csv')
data.columns=['Att1','Att2','Att3','Output']

# Plotting actual data
# sb.pairplot(data)
# plt.show()
X=data.iloc[:,0:len(data.columns.values)-1]
y=data.iloc[:,-1]
# standardizing each attribute
standardizedX=(X-X.mean())/X.std()
X=standardizedX.to_numpy().astype(float)
X=np.c_[np.ones((len(X), 1)), X]
y=y.to_numpy().reshape(-1,1).astype(float)

# plotting standardized data
# sb.pairplot(pd.concat([pd.DataFrame(standardizedX),pd.DataFrame(y,columns=['Output'])],axis=1))
# plt.show()

# Calculate YHat
def calculateY(x,theta):
    return x.dot(theta)

# MSE calculation:-
def calcuateMSE(y_org,y_cal):
    return np.sum((y_org-y_cal)**2)/len(y_org)

# ----------- Batch Gradient Descent--------------

# spliting data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
#initializing random theta
theta =np.random.randn(4,1)
# Initializing variables
eta=0.01
iterations=1000
m=len(X_train)
mses=[]
threshold=1e-3
#  epochs loop
for i in range(0,iterations):
    #calculating MSE
    yHat = calculateY(X_train, theta)
    mse = calcuateMSE(y_train, yHat)
    mses.append(mse)
    if mse < threshold:
        print('Stopping as the error is less than threshold')
        break
    # calculating Gradient
    gradeint=(2/m)*X_train.T.dot(X_train.dot(theta)-y_train)
    # calculating Theta
    theta-=eta* gradeint
y_test_res = calculateY(X_test, theta)
# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE variation using Batch Gradient Descent")
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()

# Plotting Attribute vs Regression line
for x in range(1,X_test.shape[1]):
    plt.plot(X_test[:,x], y_test_res, "r-", label="Y_Pred")
    plt.xlabel("Feature {0}".format(x))
    plt.ylabel("YHat")
    plt.title("Using Batch Gradient Descent")
    plt.legend()
    plt.show()

print("MSE for batch Gradient Descent is {0}".format(calcuateMSE(y_test,y_test_res)))

#----- Stochastic Gradient Descent----------
# spliting data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
# to pick random instance each iteration
k=1
xy=pd.concat([pd.DataFrame(X),pd.DataFrame(y)],axis=1)
#initializing random theta
theta =np.random.randn(4,1)
# Initializing variables
eta=0.01
iterations=10000
threshold=1e-3
mses=[]
#  epochs loop
for o in range(0,iterations):
    # picking random Instance each epoch
    temp=xy.sample(k)
    X_tr = temp.iloc[:, 0:4].values
    y_tr = temp.iloc[:, -1].values
    mselocal=0
    for i in range(k):
        # calculating MSE
        yHat=calculateY(X_tr[i].reshape(1,4),theta)
        mse=calcuateMSE(y_tr[i].reshape(-1,1),yHat)
        mselocal+=mse
        # calculating Gradient
        gradeint=(2)*X_tr[i].reshape(1,4).T.dot(X_tr[i].reshape(1,4).dot(theta)-y_tr[i])
        # calculating Theta
        theta-=eta*gradeint
    mses.append(mselocal)
    if mselocal<threshold:
        print("Stopping as Mse is less than threashold")
        break
    eta = eta / 1.01
y_test_res = calculateY(X_test, theta)
# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.title("MSE variation using Stochastic Gradient Descent")
plt.legend()
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()
# Plotting Attribute vs Regression line
for x in range(1,X_test.shape[1]):
    plt.plot(X_test[:,x], y_test_res, "r-", label="Y_Pred")
    plt.xlabel("Feature {0}".format(x))
    plt.ylabel("YHat")
    plt.title("Stochastic Gradient Descent")
    plt.legend()
    plt.show()
print("MSE for Stochastic Gradient Descent is {0}".format(calcuateMSE(y_test,y_test_res)))

# ----- Mini Batch Gradient Descent----------
# spliting data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
# picking a batch of 30 per each epoch
k=30
xy=pd.concat([pd.DataFrame(X),pd.DataFrame(y)],axis=1)

#initializing random theta
theta =np.random.randn(4,1)
# Initializing variables
eta=0.01
iterations=10000
threshold=1e-3
mses=[]
#  epochs loop
for o in range(0,iterations):
    # picking random Mini batch each epoch
    temp=xy.sample(k)
    X_tr = temp.iloc[:, 0:4].values
    y_tr = temp.iloc[:, -1].values
    mselocal=0
    for i in range(k):
        # calculating MSE
        yHat=calculateY(X_tr[i].reshape(1,4),theta)
        mse=calcuateMSE(y_tr[i].reshape(-1,1),yHat)
        mselocal+=mse
        # calculating Gradient
        gradeint=(2)*X_tr[i].reshape(1,4).T.dot(X_tr[i].reshape(1,4).dot(theta)-y_tr[i])
        # calculating Theta
        theta-=eta*gradeint
    mses.append(mselocal)
    if mselocal<threshold:
        print("Stopping as Mse is less than threashold")
        break
    eta = eta / 1.01
y_test_res = calculateY(X_test, theta)
# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.title("MSE variation using Mini Batch Gradient Descent")
plt.legend()
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()

# Plotting Attribute vs Regression line
for x in range(1,X_test.shape[1]):
    plt.plot(X_test[:,x], y_test_res, "r-", label="Y_Pred")
    plt.xlabel("Feature {0}".format(x))
    plt.ylabel("YHat")
    plt.title("Stochastic Gradient Descent")
    plt.legend()
    plt.show()
print("MSE for Mini Batch Gradient Descent is ",calcuateMSE(y_test, y_test_res))
# ----------- Batch Gradient Descent with L2 Regularization--------------
# spliting data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
#initializing random theta
theta =np.random.randn(4,1)
# Initializing variables
eta=0.01
iterations=1000
m=len(X_train)
mses=[]
threshold=1e-3
regressor = RidgeCV(alphas=[1e-2, 1e-1, 1e+0, 1e+1, 1e+2,0.1,0.91,0.9, 1e-13, 1e6], store_cv_values=True)
regressor.fit(X_train, y_train)
alpha=regressor.alpha_
#  epochs loop
for i in range(0,iterations):
    #calculating MSE
    yHat = calculateY(X_train, theta)
    mse = calcuateMSE(y_train, yHat)
    mses.append(mse)
    if mse < threshold:
        print('Stopping as the error is less than threshold')
        break
    # calculating Gradient
    gradeint=(2/m)*X_train.T.dot(X_train.dot(theta)-y_train)+2*alpha*theta
    theta-=eta* gradeint
y_test_res = calculateY(X_test, theta)

# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE variation using Batch Gradient Descent with L2 Regularization")
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()

#--- removing the attribute with lowest theta
L2X=np.copy(X)
L2X=np.delete(L2X,np.argmin(theta[1:]),1)
X_train, X_test, y_train, y_test = train_test_split(L2X, y, train_size=0.5)
theta =np.random.randn(3,1)
# again training with new feature set
eta=0.01
iterations=1000
m=len(X_train)
mses=[]
threshold=1e-3
for i in range(0,iterations):
    yHat = calculateY(X_train, theta)
    mse = calcuateMSE(y_train, yHat)
    mses.append(mse)
    if mse < threshold:
        print('Stopping as the error is less than threshold')
        break
    gradeint=(2/m)*X_train.T.dot(X_train.dot(theta)-y_train)
    theta-=eta* gradeint
y_test_res = calculateY(X_test, theta)

# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE variation using Batch Gradient Descent with L2 Regularization after removing a attribute")
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()

# Plotting Attribute vs Regression line
for x in range(1,X_test.shape[1]):
    plt.plot(X_test[:,x], y_test_res, "r-", label="Y_Pred")
    plt.xlabel("Feature {0}".format(x))
    plt.title("Batch Gradient Descent with L2 Regularization")
    plt.ylabel("YHat")
    plt.legend()
    plt.show()

y_test_res = calculateY(X_test, theta)
print("MSE for Batch Gradient Descent with L2 Regularization is ",calcuateMSE(y_test, y_test_res))

# ----------- Batch Gradient Descent with L1 Regularization--------------
# spliting data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
#initializing random theta
theta =np.random.randn(4,1)
# Initializing variables
eta=0.01
iterations=10000
m=len(X_train)
mses=[]
threshold=1e-3
model = LassoCV(alphas=None, cv=None, max_iter=10000)
model.fit(X_train[:,1:], y_train.reshape(-1,))
alpha=model.alpha_
#  epochs loop
for i in range(0,iterations):
    #calculating MSE
    yHat = calculateY(X_train, theta)
    mse = calcuateMSE(y_train, yHat)
    mses.append(mse)
    if mse < threshold:
        print('Stopping as the error is less than threshold')
        break
    # calculating Gradient
    gradeint=(2/m)*X_train.T.dot(X_train.dot(theta)-y_train) + alpha*np.sign(theta)
    # calculating Theta
    theta -= eta * gradeint
y_test_res = calculateY(X_test, theta)
# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE variation using Batch Gradient Descent with L1 Regularization")
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()
#--- removing the attribute with corresponidng theta values as 0
L1X=np.copy(X)
L1X=np.delete(L1X,np.where(np.round(theta,1)==0)[0],1)
X_train, X_test, y_train, y_test = train_test_split(L1X, y, train_size=0.5)
theta =np.random.randn(X_train.shape[1],1)
# again training with new feature set
eta=0.01
iterations=1000
m=len(X_train)
mses=[]
threshold=1e-3
for i in range(0,iterations):
    yHat = calculateY(X_train, theta)
    mse = calcuateMSE(y_train, yHat)
    mses.append(mse)
    if mse < threshold:
        print('Stopping as the error is less than threshold')
        break
    gradeint=(2/m)*X_train.T.dot(X_train.dot(theta)-y_train)
    theta-=eta* gradeint
y_test_res = calculateY(X_test, theta)
# ploting graph to show the variations in MSE
plt.plot(range(len(mses)), mses,label="MSE's variation")
plt.xlabel("Epoc's")
plt.ylabel("MSE")
plt.legend()
plt.title("MSE variation using Batch Gradient Descent with L1 Regularization after removing a attribute")
plt.show()
# Plotting predicted Y vs actual Y
plt.plot(range(X_test.shape[0]),y_test,"--r",label="Actual Y_test")
plt.plot(range(X_test.shape[0]),calculateY(X_test,theta),"--b",label="Predicted Y_test (YHat)")
plt.title("YHat_test vs Y_test")
plt.ylabel("Y")
plt.legend()
plt.show()
# Plotting Attribute vs Regression line
for x in range(1,X_test.shape[1]):
    plt.plot(X_test[:,x], y_test_res, "r-", label="Y_Pred")
    plt.xlabel("Feature {0}".format(x))
    plt.title("Batch Gradient Descent with L1 Regularization")
    plt.ylabel("YHat")
    plt.legend()
    plt.show()

y_test_res = calculateY(X_test, theta)
print("MSE for Batch Gradient Descent with L1 Regularization is ",calcuateMSE(y_test, y_test_res))
