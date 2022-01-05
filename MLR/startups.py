import pandas as pd                          # importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

df = pd.read_csv("50_Startups.csv")           # importing dataset

# data cleansing and eda part *********************************************************

df.nunique()                                  # checking unique values
df.duplicated().sum()                         # checking duplicate rows
df.describe()                                 # checking mean , median and sd
plt.boxplot(df.iloc[:,[0,1,2,4]])             # box plot for outliers
df = pd.get_dummies(df,drop_first = True)     # dummy column for discrete data
corr = df.corr()                              # checking correlation
df.skew()                                     # checking skewness
df.kurtosis()                                 # checking peakness
sns.pairplot(df)                              # plotting pair plots
# QQ plot for normal 
stats.probplot(df.Profit, dist = "norm", plot = pylab)
plt.show()

def norm1(i):                                 # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm = norm1(df)
# changing column names
norm = norm.rename(columns = {"R&D Spend":"rd","Administration":"ad","Marketing Spend":"ms","State_Florida":"sf","State_New York":"sn"})
         
# model building for assumption*********************************************************

ml1 = smf.ols('Profit ~ rd + ad + ms + sf + sn', data = norm).fit() 
ml1.summary()

# checling influence rows
sm.graphics.influence_plot(ml1)
norm_new = norm.drop(norm.index[[49]])          # dropping infuence row

# Preparing model                  
ml_new  = smf.ols('Profit ~ rd + ad + ms + sf + sn', data = norm).fit()   
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF

rsq_p = smf.ols('Profit ~ rd + ad + ms + sf + sn', data = norm).fit().rsquared  
vif_p = 1/(1 - rsq_p) 

rsq_rd =smf.ols('rd ~ Profit + ad + ms + sf + sn', data = norm).fit().rsquared  
vif_rd = 1/(1 - rsq_rd)

rsq_ad = smf.ols('ad ~ Profit + rd + ms + sf + sn', data = norm).fit().rsquared 
vif_ad = 1/(1 - rsq_ad) 

rsq_ms = smf.ols('ms ~ Profit + ad + rd + sf + sn', data = norm).fit().rsquared 
vif_ms = 1/(1 - rsq_ms) 

rsq_sf = smf.ols('sf ~ Profit + ad + rd + ms + sn', data = norm).fit().rsquared 
vif_sf = 1/(1 - rsq_sf) 

rsq_sn = smf.ols('sn ~ Profit + ad + rd + sf + ms', data = norm).fit().rsquared 
vif_sn = 1/(1 - rsq_sn) 

# Storing vif values in a data frame
d1 = {'Variables':['Profit', 'rd', 'ad', 'ms','sf','sn'], 'VIF':[vif_p, vif_rd, vif_ad, vif_ms ,vif_sf,vif_sn ]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As rd having high vif so dropping it

# Final model
final_ml = smf.ols('Profit ~  ad + ms + sn + sf', data = norm).fit() 
final_ml.summary() 

# Prediction
pred = final_ml.predict(norm)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = norm.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

# Splitting the data into train and test data 

norm_train, norm_test = train_test_split(norm, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Profit ~  ad + ms + sn + sf', data = norm_train).fit()

# prediction on test data set 
test_pred = model_train.predict(norm_test)

# test residual values 
test_resid = test_pred - norm_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(norm_train)

# train residual values 
train_resid  = train_pred - norm_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

