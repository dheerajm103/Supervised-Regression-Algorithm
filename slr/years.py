import pandas as pd                                      # importing libraries
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("Salary_Data.csv")                # importing dataset
df

# datacleansing and eda part****************************************************************************************

df.info()                                               # checking for datatypes and null values
df.describe()                                           # for mean , median and sd
plt.boxplot(df)                                         # box plot for outliers

plt.hist(df.salary)                                         # plotting histogram
plt.hist(df.years)
plt.scatter(df.years,df.salary , color = "green")               # plotting scatter plot

df.corr()                                               # checking correlation
df.skew()                                               # cheking skewness
df.kurtosis()                                           # for kurtosis
df.cov()                                                # for covariance

# finding best model*******************************************************************************************

# Simple Linear Regression
model = smf.ols('salary ~ years', data = df).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(df['years']))

# Regression Line
plt.scatter(df.years, df.salary)
plt.plot(df.years, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = df.salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# log transformation

plt.scatter(x = np.log(df['years']), y = df['salary'], color = 'brown')
np.corrcoef(np.log(df.years), df.salary) 


model2 = smf.ols('salary ~ np.log(years)', data = df).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(df['years']))

# Regression Line
plt.scatter(np.log(df.years), df.salary)
plt.plot(np.log(df.years), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = df.salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation

plt.scatter(x = df['years'], y = np.log(df['salary']), color = 'orange')
np.corrcoef(df.years, np.log(df.salary)) #correlation

model3 = smf.ols('np.log(salary) ~ years', data = df).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(df['years']))
pred3_s = np.exp(pred3)
pred3_s

# Regression Line
plt.scatter(df.years, np.log(df.salary))
plt.plot(df.years, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = df.salary - pred3_s
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


# Polynomial transformation

model4 = smf.ols('np.log(salary) ~ years + I(years*years)', data = df).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(df))
pred4_s = np.exp(pred4)
pred4_s

# Regression line

poly_reg = PolynomialFeatures(degree = 2)
X = df.iloc[:, [0]].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(df.years, np.log(df.salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = df.salary - pred4_s
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

finalmodel = smf.ols('np.log(salary) ~ years + I(years*years)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))

# Model Evaluation on Test data
test_res = test.salary - test_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))

# Model Evaluation on train data
train_res = train.salary - train_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
