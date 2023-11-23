import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVR

# reading the data_set
dataset = pd.read_csv('/Users/zakariadarwish/Desktop/support_vector_regression/Position_Salaries.csv')
# feeding the dependant & independant variables
# where X is dep & y is indep
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# transform the shape of 1D vector into 2D Vector for feature scaling
y = y.reshape(len(y),1)


sc_X = StandardScaler()
sc_y = StandardScaler()

# apply feature scaling on dep. and indep. varaibles (i.e X,y)
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# applying SVR regressor model 
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# predicting a specific value after ransformation and reshaping it ... 
y_predict = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))


# plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
# plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()