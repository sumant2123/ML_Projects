
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("HamiltonCases.csv", delimiter=',')
X = data[:,0:4]
y = data[:,4]

# // splitting dataset//
x_train=X[:250]
y_train=y[:250]

x_test=X[250:]
y_test=y[250:]

# //LINEAR REGRESSION //


from sklearn.linear_model import LinearRegression

lin=LinearRegression()
lin.fit(x_train,y_train)
y_pred = lin.predict(x_test)
print("Coefficients of linear regression are {}".format(lin.coef_))
print("Intercept of linear regression is {}".format(lin.intercept_))

# //mean squared error//

from sklearn.metrics import mean_squared_error

print("Mean Squared Error for test data is {}".format(mean_squared_error(y_test,y_pred)))
print("Mean Squared Error for training data is {}".format(mean_squared_error(y_test,lin.predict(x_test))))

# //NORMAL EQUATIONS//

xb=np.c_[np.ones((250,1)),x_train]
thetabest=np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y_train)
print(thetabest)
# //BATCH GRADIENT DESCENT//

eta=0.1
n_iterations=6
m=250
theta=np.zeros((5,1))

for iterations in range(n_iterations):
    gradients=2/m*xb.T.dot(xb.dot(theta)-y)
    theta=theta-eta*gradients
print("theta values are using gradient descent {}".format(thetabest))

# //STOCHASTIC GRADIENT DESCENT (L1 penalty)//

from sklearn.linear_model import SGDRegressor

lin1=SGDRegressor(penalty="l1")
lin1.fit(x_train,y_train)
# print(lin1.score(x_test,y_test))
y_pred1=lin1.predict(x_test)
plt.plot(y_test, y_pred1,"b.")
print("Intercept is {}".format(lin1.intercept_))
print("Coefficients are {}".format(lin1.coef_))

# //mean squared error//
from sklearn.metrics import mean_squared_error
print("Mean Squared Error is {}".format(mean_squared_error(y_test,y_pred1)))
print("Mean Squared Error for training data is {}".format(mean_squared_error(y_test,lin1.predict(x_test))))

# //STOCHASTIC GRADIENT DESCENT (L2 penalty)

from sklearn.linear_model import SGDRegressor

lin2=SGDRegressor(penalty="l2")
lin2.fit(x_train,y_train)
y_pred2=lin2.predict(x_test)
plt.plot(y_test, y_pred2,"b.")
print("Intercept is {}".format(lin2.intercept_))
print("Coefficients are {}".format(lin2.coef_))

# //mean squared error//
from sklearn.metrics import mean_squared_error
print("Mean Squared Error is {}".format(mean_squared_error(y_test,y_pred2)))
print("Mean Squared Error for training data is {}".format(mean_squared_error(y_test,lin2.predict(x_test))))

# // PLOTS//

import matplotlib.pyplot as plt

plt.plot(y_test,y_pred,label='LINEAR REGRESSION')
plt.legend(loc='upper right')
# plt.show()

plt.plot(y_test,y_pred1,label='STOCHASTIC GD L1')
plt.legend(loc='upper right')
# plt.show()

plt.plot(y_test,y_pred2,label='STOCHASTIC GD L2')
plt.legend(loc='upper right')
plt.show()
