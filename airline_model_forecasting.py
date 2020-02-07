############# MODEL_BASED #################
import pandas as pd
air = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/Airlines+Data.csv")
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = air["Month"][0]
p[0:3]
air['months']= 0

for i in range(96):
    p = air["Month"][i]
    air['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(air['months']))
air1 = pd.concat([air,month_dummies],axis = 1)

air1["t"] = np.arange(1,97)

air1["t_squared"] = air1["t"]*air1["t"]
air1.columns
air1["log_Pass"] = np.log(air1["Passengers"])
air1.rename(columns={"Passengers ": 'Passengers'}, inplace=True)
air1.Passengers.plot()
Train = air1.head(77)
Test = air1.tail(19)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Pass~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Pass~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Pass~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 

predict_data = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/Predict_new.csv")
model_full = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=air1).fit()

pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new

predict_data["forecasted_Passengers"] = pd.Series(pred_new)
##predict_data = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/Predict_new.csv")
#model_full = smf.ols('log_pas~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=airline1).fit()
#pred_full=model_full.predict(airline1)
#airline1["pred_full"]=pred_full
#residuals=pd.DataFrame(np.array(airline1["Passengers"]-np.array(pred_full)))
#pred_new  = pd.Series(model_full.predict(predict_data))
#pred_new

#predict_data["forecasted_passengers"] = pred_new

############# DATA_DRIVEN#####################################################################3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing #SES
from statsmodels.tsa.holtwinters import Holt # holt exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime, time

air = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/Airlines+Data.csv")
air.shape
air.rename(columns={"Passengers ('000)":"Passengers"}, inplace=True)
air.index = pd.to_datetime(air.Month,format="%b-%y")
air.Passengers.plot()
air["Date"] = pd.to_datetime(air.Month,format="%b-%y")
air["month"] = air.Date.dt.strftime("%b") # month extraction
air["month"] = air.Date.dt.strftime("%b") # month extraction
air["year"] = air.Date.dt.strftime("%Y") # year extraction
heatmap_y_month = pd.pivot_table(data=air,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")
# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=air)
sns.boxplot(x="year",y="Passengers",data=air)
sns.lineplot(x="year",y="Passengers",hue="month",data=air)

air.Passengers.plot(label="org")

for i in range(2,96,5):
    air["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(air.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(air.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(air.Passengers,lags=10)
tsa_plots.plot_pacf(air.Passengers)

# air.index.freq = "MS" 
# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = air.head(133)
Test = air.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers) # 7.846321

# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers) # 7.261176729658341



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) # 4.500954



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers) # 4.109309


# Lets us use auto_arima from p
from pyramid.arima import auto_arima
auto_arima_model = auto_arima(Train["Passengers"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
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
MAPE(pred_test,Test.Passengers)  # 8.75353


# Using Sarimax from statsmodels 
# As we do not have automatic function in indetifying the 
# best p,d,q combination 
# iterate over multiple combinations and return the best the combination
# For sarimax we require p,d,q and P,D,Q 
combinations_l = list(product(range(1,7),range(2),range(1,7)))
combinations_u = list(product(range(1,7),range(2),range(1,7)))
m =12 

results_sarima = []
best_aic = float("inf")

for i in combinations_l:
    for j in combinations_u:
        try:
            model_sarima = sm.tsa.statespace.SARIMAX(Train["Passengers"],
                                                     order = i,seasonal_order = j+(m,)).fit(disp=-1)
        except:
            continue
        aic = model_sarima.aic
        if aic < best_aic:
            best_model = model_sarima
            best_aic = aic
            best_l = i
            best_u = j
        results_sarima.append([i,j,model_sarima.aic])


result_sarima_table = pd.DataFrame(results_sarima)
result_sarima_table.columns = ["paramaters_l","parameters_j","aic"]
result_sarima_table = result_sarima_table.sort_values(by="aic",ascending=True).reset_index(drop=True)

best_fit_model = sm.tsa.statespace.SARIMAX(Train["Passengers"],
                                                     order = (1,1,1),seasonal_order = (1,1,1,12)).fit(disp=-1)
best_fit_model.summary()
best_fit_model.aic # 1350.449
srma_pred = best_fit_model.predict(start = Test.index[0],end = Test.index[-1])
air["srma_pred"] = srma_pred

# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")
plt.plot(pred_hwe_mul_add.index,srma_pred,label="Auto_Sarima",color="purple")
plt.legend(loc='best')

# Models and their MAPE values
model_mapes = pd.DataFrame(columns=["model_name","mape"])
model_mapes["model_name"] = ["]
# Visualizing the ACF and PACF plots for errors 
