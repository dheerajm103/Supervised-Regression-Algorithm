import pandas as pd                          # importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

df = pd.read_csv("Computer_Data.csv")           # importing dataset

# data cleansing and eda part *********************************************************

df = df.drop(["index"], axis = 1)
df.nunique()                                  # checking unique values
df.duplicated().sum()                         # checking duplicate rows
df.info()
df.describe()                                 # checking mean , median and sd
df = pd.get_dummies(df,drop_first = True)     # dummy column for discrete data
plt.boxplot(df)                               # box plot for outliers
corr = df.corr()                              # checking correlation
df.skew()                                     # checking skewness
df.kurtosis()                                 # checking peakness
sns.pairplot(df)                              # plotting pair plots
# QQ plot for normal 
stats.probplot(df.price, dist = "norm", plot = pylab)
plt.show()

def norm1(i):                                 # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm = norm1(df)
         
# model building for assumption*********************************************************

ml1 = smf.ols('price ~ speed + hd + screen + ram + ads + trend + cd_yes + multi_yes + premium_yes', data = norm).fit() 
ml1.summary()

# checling influence rows
sm.graphics.influence_plot(ml1)

# Prediction
pred = ml1.predict(norm)

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = norm.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


# Splitting the data into train and test data 

norm_train, norm_test = train_test_split(norm, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + screen + ram + ads + trend + cd_yes + multi_yes + premium_yes', data = norm_train).fit()

# prediction on test data set 
test_pred = model_train.predict(norm_test)

# test residual values 
test_resid = test_pred - norm_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(norm_train)

# train residual values 
train_resid  = train_pred - norm_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

