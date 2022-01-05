import pandas as pd                                      # importing libraries
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("calories_consumed.csv")                # importing dataset
df

# datacleansing and eda part****************************************************************************************

df.info()                                               # checking for datatypes and null values
df.describe()                                           # for mean , median and sd
df.duplicated().sum()
plt.boxplot(df)                                         # box plot for outliers

plt.hist(df.wg)                                         # plotting histogram
plt.hist(df.c)
plt.scatter(df.wg,df.c , color = "green")               # plotting scatter plot

df.corr()                                               # checking correlation
df.skew()                                               # cheking skewness
df.kurtosis()                                           # for kurtosis
df.cov()                                                # for covariance

# finding best model*******************************************************************************************

# Simple Linear Regression
model = smf.ols('c ~ wg', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['wg']))

# Regression Line
plt.scatter(df.wg, df.c)
plt.plot(df.wg, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.c - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# log transformation

plt.scatter(x = np.log(df['wg']), y = df['c'], color = 'brown')
np.corrcoef(np.log(df.wg), df.c) 


model2 = smf.ols('c ~ np.log(wg)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['wg']))

# Regression Line
plt.scatter(np.log(df.wg), df.c)
plt.plot(np.log(df.wg), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.c - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation

plt.scatter(x = df['wg'], y = np.log(df['c']), color = 'orange')
np.corrcoef(df.wg, np.log(df.c)) #correlation

model3 = smf.ols('np.log(c) ~ wg', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['wg']))
pred3_c = np.exp(pred3)
pred3_c

# Regression Line
plt.scatter(df.wg, np.log(df.c))
plt.plot(df.wg, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.c - pred3_c
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# Polynomial transformation

model4 = smf.ols('np.log(c) ~ wg + I(wg*wg)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_c = np.exp(pred4)
pred4_c

# Regression line

poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(df.wg, np.log(df.c))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.c - pred4_c
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE*******************************************************************************
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# model building on best model*********************************************************************************
# splitting dataset
train, test = train_test_split(df, test_size = 0.2)

finalmodel = smf.ols('c ~ wg ', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.c - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.c - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
