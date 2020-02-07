import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots

cocacolaData = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/Forecasting/CocaCola_Sales_Rawdata.csv")
cocacolaData.drop_duplicates(keep='first',inplace=True)
cocacolaData.isnull()
cocacolaData.isnull().sum()

cocacolaData.shape
cocacolaData.head(13)
cocacolaData.columns
cocacolaData.Sales.plot()

#plt.hist(cocacolaData.Quarter)
plt.hist(cocacolaData.Sales)

sns.boxplot(cocacolaData.Sales)
sns.pairplot((cocacolaData),hue="Sales")

#corr = cocacolaData.corr()
#corr
#sns.heatmap((corr),annot=True)

np.var(cocacolaData)
np.std(cocacolaData)

##################### MODEL BASED APPROACH #####################

cocacolaData.Quarter[0][0:2]
cocacolaData['quater_name']=0
for i in range(0,42):
    p = cocacolaData.Quarter[i]
    cocacolaData['quater_name'][i]= p[0:2]
quater_dummies = pd.get_dummies(cocacolaData['quater_name'])
cocacolaData_dumm = pd.concat([cocacolaData,quater_dummies],axis=1)

cocacolaData_dumm['t'] = np.arange(1,43)
cocacolaData_dumm["t_squared"] = cocacolaData_dumm["t"]*cocacolaData_dumm["t"]
cocacolaData_dumm["log_Sales"] = np.log(cocacolaData_dumm["Sales"])
cocacolaData_dumm = cocacolaData_dumm.drop('Quarter',axis=1)

Train = cocacolaData_dumm.head(38)
Test = cocacolaData_dumm.tail(4)

import statsmodels.formula.api as smf
linear_model = smf.ols('Sales ~ t',data = Train).fit() # Linear
linear_model.summary()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

Exp = smf.ols('log_Sales~t',data=Train).fit() # Exponential
Exp.summary()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

Quad = smf.ols('Sales~t+t_squared',data=Train).fit() # Quadratic
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit() # Additive seasonality
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit() # Additive Seasonality Quadratic
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit() # Multiplicative Seasonality
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit() # Multiplicative Additive Seasonality
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test[['t','Q1','Q2','Q3','Q4']]))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea

Mul_Add_qa_sea = smf.ols('log_Sales~t+t_squared+Q1+Q2+Q3+Q4',data = Train).fit() # Multiplicative Additive Quadratic Seasonality
pred_Mult_add_qa_sea = pd.Series(Mul_Add_sea.predict(Test[['t',"t_squared",'Q1','Q2','Q3','Q4']]))
rmse_Mult_add_qa_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_qa_sea)))**2))
rmse_Mult_add_qa_sea

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","Mul_Add_qa_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_add_qa_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

plt.plot(data['MODEL'],data['RMSE_Values'])
plt.scatter(data['MODEL'],data['RMSE_Values'])

####################### DATA DRIVEN APPROACH #########################

cocacolaData.Quarter[0][3:]
cocacolaData['quater_name']=0
cocacolaData['year_name']=0
for i in range(0,42):
    p = cocacolaData.Quarter[i]
    cocacolaData['quater_name'][i]= p[0:2]
    cocacolaData['year_name'][i]= p[3:]

heatmap_y_month = pd.pivot_table(data=cocacolaData,values="Sales",index="year_name",columns="quater_name",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

sns.boxplot(x="quater_name",y="Sales",data=cocacolaData)
sns.boxplot(x="year_name",y="Sales",data=cocacolaData)
sns.lineplot(x="year_name",y="Sales",hue="quater_name",data=cocacolaData)

cocacolaData.Sales.plot(label="Sales")
for i in range(2,24,6):
    cocacolaData["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

tsa_plots.plot_acf(cocacolaData.Sales,lags=10)
tsa_plots.plot_pacf(cocacolaData.Sales)

Train = cocacolaData.head(38)
Test = cocacolaData.tail(4)

def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales)

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)

tendArr = ["add", "mul", "additive", "multiplicative"]
sesonalArr = ["add", "mul", "additive", "multiplicative"]

mapeArr = [];
for i in tendArr:
    for j in sesonalArr:
        hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal=j,trend=i,seasonal_periods=12,damped=True).fit()
        pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
        mapeArr.append([i,j,MAPE(pred_hwe_add_add,Test.Sales)])
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) # 8.820771292784366
