import pandas as pd                          # importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

df = pd.read_csv("ToyotaCorolla.csv")           # importing dataset

# data cleansing and eda part *********************************************************
df = df.iloc[:,[1,2,5,7,11,12,14,15,16]]
df.nunique()                                  # checking unique values
df.duplicated().sum()                         # checking duplicate rows
df = df.drop_duplicates()
df.info()
df.describe()                                 # checking mean , median and sd
plt.boxplot(df)                               # box plot for outliers
corr = df.corr()                              # checking correlation
df.skew()                                     # checking skewness
df.kurtosis()                                 # checking peakness
sns.pairplot(df)                              # plotting pair plots
# QQ plot for normal 
stats.probplot(df.Price, dist = "norm", plot = pylab)
plt.show()

def norm1(i):                                 # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm = norm1(df)

# model building for assumption*********************************************************

ml1 = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + Qt + weight', data = norm).fit() 
ml1.summary()

# checling influence rows
sm.graphics.influence_plot(ml1)
norm_new = norm.drop(norm.index[[80]])          # dropping infuence row

# Preparing model                  
ml_new  = smf.ols('Price ~ Age + Doors + KM + HP + cc  + Gears + Qt + weight', data = norm_new).fit()   
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF

rsq_p = smf.ols('Price ~ Age + KM + HP + cc + Doors + Gears + Qt + weight', data = norm).fit().rsquared  
vif_p = 1/(1 - rsq_p) 

rsq_a =smf.ols('Age ~ Price + KM + HP + cc + Doors + Gears + Qt + weight', data = norm).fit().rsquared  
vif_a = 1/(1 - rsq_a)

rsq_k = smf.ols('KM ~ Age + Price + HP + cc + Doors + Gears + Qt + weight', data = norm).fit().rsquared 
vif_k = 1/(1 - rsq_k) 

rsq_h = smf.ols('HP ~ Age + KM + Price + cc + Doors + Gears + Qt + weight', data = norm).fit().rsquared 
vif_h = 1/(1 - rsq_h) 

rsq_cc = smf.ols('cc ~ Age + KM + HP + Price + Doors + Gears + Qt + weight', data = norm).fit().rsquared 
vif_cc = 1/(1 - rsq_cc) 

rsq_d = smf.ols('Doors ~ Age + KM + HP + cc + Price + Gears + Qt + weight', data = norm).fit().rsquared 
vif_d = 1/(1 - rsq_d) 

rsq_g = smf.ols('Gears ~ Age + KM + HP + cc + Doors + Price + Qt + weight', data = norm).fit().rsquared 
vif_g = 1/(1 - rsq_g) 

rsq_q = smf.ols('Qt ~ Age + KM + HP + cc + Doors + Gears + Price + weight', data = norm).fit().rsquared 
vif_q = 1/(1 - rsq_q) 

rsq_w = smf.ols('weight ~ Age + KM + HP + cc + Doors + Gears + Qt + Price', data = norm).fit().rsquared 
vif_w = 1/(1 - rsq_w) 

# Storing vif values in a data frame
d1 = {'Variables':['Price', 'Age', 'km', 'hp','cc','Doors','Gears','Qt','weight'], 'VIF':[vif_p, vif_a, vif_k, vif_h ,vif_cc,vif_d,vif_g,vif_q,vif_w]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Final model
final_ml = smf.ols('Price ~ Age + KM + HP + cc  + Gears + Qt + weight', data = norm_new).fit() 
final_ml.summary() 

# Prediction
pred = final_ml.predict(norm_new)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = norm_new.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

# Splitting the data into train and test data 

norm_train, norm_test = train_test_split(norm_new, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age + KM + HP + cc  + Gears + Qt + weight', data = norm_train).fit()

# prediction on test data set 
test_pred = model_train.predict(norm_test)

# test residual values 
test_resid = test_pred - norm_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(norm_train)

# train residual values 
train_resid  = train_pred - norm_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

