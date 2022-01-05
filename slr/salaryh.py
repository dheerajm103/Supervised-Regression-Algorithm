import pandas as pd                                      # importing libraries
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("emp_data.csv")                # importing dataset
df

# datacleansing and eda part****************************************************************************************

df.info()                                               # checking for datatypes and null values
df.describe()                                           # for mean , median and sd
df.duplicated().sum()
plt.boxplot(df)                                         # box plot for outliers

plt.hist(df.salary)                                         # plotting histogram
plt.hist(df.churn)
plt.scatter(df.churn,df.salary , color = "green")               # plotting scatter plot

df.corr()                                               # checking correlation
df.skew()                                               # cheking skewness
df.kurtosis()                                           # for kurtosis
df.cov()                                                # for covariance

# finding best model*******************************************************************************************

# Simple Linear Regression
model = smf.ols('churn ~ salary', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['salary']))

# Regression Line
plt.scatter(df.salary, df.churn)
plt.plot(df.salary, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.churn - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# log transformation

plt.scatter(x = np.log(df['salary']), y = df['churn'], color = 'brown')
np.corrcoef(np.log(df.salary), df.churn) 


model2 = smf.ols('churn ~ np.log(salary)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['salary']))

# Regression Line
plt.scatter(np.log(df.salary), df.churn)
plt.plot(np.log(df.salary), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.churn - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation

plt.scatter(x = df['salary'], y = np.log(df['churn']), color = 'orange')
np.corrcoef(df.salary, np.log(df.churn)) #correlation

model3 = smf.ols('np.log(churn) ~ salary', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['salary']))
pred3_c = np.exp(pred3)
pred3_c

# Regression Line
plt.scatter(df.salary, np.log(df.churn))
plt.plot(df.salary, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.churn - pred3_c
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# Polynomial transformation

model4 = smf.ols('np.log(churn) ~ salary + I(salary*salary)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_c = np.exp(pred4)
pred4_c

# Regression line

poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, [0]].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(df.salary, np.log(df.churn))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.churn - pred4_c
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

finalmodel = smf.ols('np.log(churn) ~ salary + I(salary*salary)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.churn - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.churn - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
