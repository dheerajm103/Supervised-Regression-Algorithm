import pandas as pd                          # importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

df = pd.read_csv("Avacado_Price.csv")           # importing dataset

# data cleansing and eda part *********************************************************

df.nunique()                                  # checking unique values
df.duplicated().sum()                         # checking duplicate rows
df.describe()                                 # checking mean , median and sd
df.info()
plt.boxplot(df.iloc[:,0:8])                    # box plot for outliers
df = df.drop(["region","year","type"], axis = 1)
corr = df.corr()                              # checking correlation
df.skew()                                     # checking skewness
df.kurtosis()                                 # checking peakness
sns.pairplot(df)                              # plotting pair plots
# QQ plot for normal 
stats.probplot(df.AveragePrice, dist = "norm", plot = pylab)
plt.show()

def norm1(i):                                 # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm = norm1(df)
# changing column names
norm = norm.rename(columns = {"AveragePrice":"ap","Total_Volume":"tv","tot_ava1":"ta1","tot_ava2":"ta2","tot_ava3":"ta3","Total_Bags":"tb","Small_Bags":"sb","Large_Bags":"lb","XLarge Bags":"xlb"})
         
# model building for assumption*********************************************************

ml1 = smf.ols('ap ~ tv + ta1 + ta2 + ta3 + tb + sb + lb + xlb', data = norm).fit() 
ml1.summary()

# checling influence rows
sm.graphics.influence_plot(ml1)
norm_new = norm.drop(norm.index[[17468]])          # dropping infuence row

# Preparing model                  
ml_new  = smf.ols('ap ~ tv + ta1 + ta2 + ta3 + tb + sb + lb + xlb', data = norm_new).fit()   
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF

rsq_ap = smf.ols('ap ~ tv + ta1 + ta2 + ta3 + tb + sb + lb + xlb', data = norm).fit().rsquared  
vif_ap = 1/(1 - rsq_ap) 

rsq_tv =smf.ols('tv ~ ap + ta1 + ta2 + ta3 + tb + sb + lb + xlb', data = norm).fit().rsquared  
vif_tv = 1/(1 - rsq_tv)

rsq_ta1 = smf.ols('ta1 ~ ap + tv + ta2 + ta3 + tb + sb + lb + xlb', data = norm).fit().rsquared 
vif_ta1 = 1/(1 - rsq_ta1) 

rsq_ta2 = smf.ols('ta2 ~ ap + tv + ta1 + ta3 + tb + sb + lb + xlb', data = norm).fit().rsquared 
vif_ta2 = 1/(1 - rsq_ta2) 

rsq_ta3 = smf.ols('ta3 ~ ap + tv + ta2 + ta1 + tb + sb + lb + xlb', data = norm).fit().rsquared 
vif_ta3 = 1/(1 - rsq_ta3) 

rsq_tb = smf.ols('tb ~ ap + tv + ta2 + ta3 + ta1 + sb + lb + xlb', data = norm).fit().rsquared 
vif_tb = 1/(1 - rsq_tb) 

rsq_sb = smf.ols('sb ~ ap + tv + ta2 + ta3 + ta1 + tb + lb + xlb', data = norm).fit().rsquared 
vif_sb = 1/(1 - rsq_sb) 

rsq_lb = smf.ols('lb ~ ap + tv + ta2 + ta3 + ta1 + sb + tb + xlb', data = norm).fit().rsquared 
vif_lb = 1/(1 - rsq_lb) 

rsq_xlb = smf.ols('xlb ~ ap + tv + ta2 + ta3 + ta1 + sb + lb + tb', data = norm).fit().rsquared 
vif_xlb = 1/(1 - rsq_xlb) 

# Storing vif values in a data frame
d1 = {'Variables':['ap', 'tv', 'ta1', 'ta2','ta3','tb','sb','lb','xlb'], 'VIF':[vif_ap, vif_tv, vif_ta1, vif_ta2 ,vif_ta3,vif_tb,vif_sb,vif_lb,vif_xlb]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As tb having high vif so dropping it

# Final model
final_ml = smf.ols('ap ~ tv + ta1 + ta2 + ta3  + sb + lb + xlb', data = norm).fit() 
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
sns.residplot(x = pred, y = norm.ap, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

# Splitting the data into train and test data 

norm_train, norm_test = train_test_split(norm, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('ap ~ tv + ta1 + ta2 + ta3 + tb + sb + lb + xlb', data = norm_train).fit()

# prediction on test data set 
test_pred = model_train.predict(norm_test)

# test residual values 
test_resid = test_pred - norm_test.ap
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(norm_train)

# train residual values 
train_resid  = train_pred - norm_train.ap
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

