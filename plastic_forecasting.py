#################### MODEL_BASED ######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plastic= pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/PlasticSales.csv")
plastic.info()
plastic.describe()
plastic.head()
plastic.isnull().sum()
plastic.dtypes
plastic.shape

# Histogram 
plt.hist(plastic['Sales'], color='red');plt.title('Histogram of Sales')


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p=plastic['Month'][0]
p[0:3]
plastic['months']=0

for i in range(60):
    p=plastic["Month"][i]
    plastic["months"][i]=p[0:3]
month_dummies=pd.DataFrame(pd.get_dummies(plastic["months"]))
plastic1=pd.concat([plastic,month_dummies],axis=1)

# Drop the "Month" column
plastic1.drop('Month', axis=1, inplace=True)

plastic1["t"]=np.arange(1,61)
plastic1["t_sqr"]=plastic1["t"]*plastic1["t"]
plastic1["log_sales"]=np.log(plastic1["Sales"])
plastic1.columns
plastic1.Sales.plot()

train= plastic1.head(45)
test=plastic1.tail(12)

import statsmodels.formula.api as smf 

###  Linear ##################################

linear_model= smf.ols("Sales~t", data=train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(test["t"])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
rmse_linear ## 254.948

### Exponential ##########################

exp_model= smf.ols("log_sales~t", data=train).fit()
pred_exp=pd.Series(exp_model.predict(pd.DataFrame(test["t"])))
rmse_exp = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_exp)))**2))
rmse_exp ## 262.11

#### Quadratic #######################

quad_model=smf.ols("Sales~t+t_sqr", data=train).fit()
pred_quad=pd.Series(quad_model.predict(test[["t","t_sqr"]]))
rmse_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_quad))**2))
rmse_quad ## 296.46

#### Additive Seasonality ########################

add_sea=smf.ols("Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov", data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea ## 240.15

#### Additive Seasonality Quadratic #############

add_sea_Quad = smf.ols('Sales~t+t_sqr+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sqr']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  ## 163.35

#### Multiplicative Seasonality ###################

mult_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_mult_sea = pd.Series(mult_sea.predict(test))
rmse_mult_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_sea)))**2))
rmse_mult_sea ## 244.41

#### Multiplicative Additive Seasonality ###################

mult_add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_mult_add_sea = pd.Series(mult_add_sea.predict(test))
rmse_mult_add_sea = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mult_add_sea)))**2))
rmse_mult_add_sea ## 135.11

####### Testing #############
data={"Model":pd.Series(["rmse_linear","rmse_exp","rmse_quad","rmse_add_sea","rmse_add_sea_quad","rmse_mult_sea","rmse_mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mult_sea,rmse_mult_add_sea])}
RMSE_table = pd.DataFrame(data)
RMSE_table

# so Multiplicative_additive_seasonality has the least RMSE value among the models.i.e 135.11

################### DATA_DRIVEN####################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
#from sm.tsa.statespace import sa
plastic = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/PlasticSales.csv")
   
# Converting the normal index of plastic to time stamp 
plastic.index = pd.to_datetime(plastic.Month,format="%b-%y")

plastic.Sales.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
plastic["Date"] = pd.to_datetime(plastic.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

plastic["month"] = plastic.Date.dt.strftime("%b") # month extraction
#Amtrak["Day"] = Amtrak.Date.dt.strftime("%d") # Day extraction
#Amtrak["wkday"] = Amtrak.Date.dt.strftime("%A") # weekday extraction
plastic["year"] = plastic.Date.dt.strftime("%Y") # year extraction

# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=plastic,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Sales",data=plastic)
sns.boxplot(x="year",y="Sales",data=plastic)
# sns.factorplot("month","Ridership",data=Amtrak,kind="box")

# Line plot for Ridership based on year  and for each month
sns.lineplot(x="year",y="Sales",hue="month",data=plastic)


# moving average for the time series to understand better about the trend character in plastic
plastic.Sales.plot(label="org")
for i in range(2,24,6):
    plastic["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic.Sales,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(plastic.Sales,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(plastic.Sales,lags=12)
tsa_plots.plot_pacf(plastic.Sales,lags=12)

# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = plastic.head(48)
Test = plastic.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

Train["Sales"] = Train["Sales"].astype('double')
# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) #17.04 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) # 101.98


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales) # 14.42

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales) # 15.01

# Lets us use auto_arima from p
from pyramid.arima import auto_arima
auto_arima_model = auto_arima(Train["Sales"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=4,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
                
            
auto_arima_model.summary() # SARIMAX(1, 1, 1)x(0, 1, 1, 12)
# AIC ==> 1348.728
# BIC ==> 1362.665

# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )

# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=12))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Sales)  # 12.72

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(plastic.Sales,order=(1,1,0)).fit(transparams=True)
forecasterrors=model.forecast(steps=12)[0]#it will give the next 12 values
