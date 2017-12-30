## Import Libraries

import numpy as np
import pandas as pd
import sklearn
import scipy

## Import Data

new_data=pd.read_csv("new_data.csv")


print("Preprocessing is Started!! Please Wait.... \n")

##### Data Preprocessing #####

	##### Missing Values #####

new_data.libor_rate=new_data.libor_rate.fillna(new_data.libor_rate.mean())
new_data.indicator_code=new_data.indicator_code.fillna(-999)
new_data.hedge_value=new_data.hedge_value.fillna(False)
new_data.status=new_data.status.fillna(False)

	##### Applying Label Encoding #######

pf_cat={"A":0,"B":1,"C":2,"D":3,"E":4}
cc={"M":0,"N":1,"T":2,"U":3,"Z":4}
typ={"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7}
hedge={True:1,False:0}
stat={True:1,False:0}
indi={True:1,-999:-999}

		### PF Category
new_data.pf_category=new_data.pf_category.apply(lambda x: pf_cat[x])

		## Country Code
new_data.country_code=new_data.country_code.apply(lambda x: cc[x])

		## Type
new_data.type=new_data.type.apply(lambda x: typ[x])

		## Status
new_data.status=new_data.status.apply(lambda x: stat[x])

		## Indicator 
new_data.indicator_code=new_data.indicator_code.apply(lambda x: indi[x])

		## Hedge value
new_data.hedge_value=new_data.hedge_value.apply(lambda x: hedge[x])


##Dealing with Date-Time Objects
		##Start date
new_data.start_date=pd.to_datetime(new_data.start_date,format='%Y%m%d', errors='ignore')
		## Creation Date
new_data.creation_date=pd.to_datetime(new_data.creation_date,format='%Y%m%d', errors='ignore')
	## Sell date
test.sell_date=pd.to_datetime(test.sell_date,format='%Y%m%d', errors='ignore')


##### Currency Converter #####

def Currency_Converter(data):
    cur=data[0]
    x=data[1]
    if cur=="USD":
        return(x)
    elif cur=="GBP":
        return(x*1.3)
    elif cur=="EUR":
        return (x*1.2)
    elif cur=="CHF":
        return(x)
    elif cur=="JPY":
        return(x*0.008)
	#### Applying Currency Converter to Columns #####

		## Sold
new_data.sold=new_data[["currency","sold"]].apply(Currency_Converter,axis=1)
		##Bought
new_data.bought=new_data[["currency","bought"]].apply(Currency_Converter,axis=1)	
		## Euribor_Rate
new_data.euribor_rate=new_data[["currency","euribor_rate"]].apply(Currency_Converter,axis=1)	
	## Libor_Rate
new_data.libor_rate=new_data[["currency","libor_rate"]].apply(Currency_Converter,axis=1)

## Currency
def cur(data):
    if data=="USD":
        return(1)
    else:
        return(0)
new_data.currency=new_data.currency.apply(lambda x: cur(x))

	#### Special Treatment for Sold and Bought ######

from scipy.stats.mstats import winsorize
winsorize(a=new_data.sold,limits=0.1,inplace=True)
winsorize(a=new_data.bought,limits=0.1,inplace=True)


##### New Feature Generation #####

new_data["start_day"]=(new_data.creation_date.dt.year-new_data.start_date.dt.year)*365+(new_data.creation_date.dt.month-new_data.start_date.dt.month)*30+(new_data.creation_date.dt.day-new_data.start_date.dt.day)

new_data["sell_day"]=(new_data.sell_date.dt.year-new_data.start_date.dt.year)*365+(new_data.sell_date.dt.month-new_data.start_date.dt.month)*30+(new_data.sell_date.dt.day-new_data.start_date.dt.day)

new_data["creation_day"]=(new_data.sell_date.dt.year-new_data.creation_date.dt.year)*365+(new_data.sell_date.dt.month-new_data.creation_date.dt.month)*30+(new_data.sell_date.dt.day-new_data.creation_date.dt.day)


##### Rearranging Columns

new_data=new_data[['pf_category','country_code','euribor_rate','currency',
       'libor_rate', 'bought',"sold",'indicator_code','type', 'hedge_value',"status",'start_day','sell_day',
       'creation_day','return', ]]


### Exporting into CSV Files

new_data.to_csv("Cleaned_New_Data.csv",index=False)

print("Preprocessing is Done!! Output File : Cleaned_New_Data.csv \n")


"""


#### Modelling #######

new_data=pd.read_csv("Cleaned_New_Data.csv")

X=new_data.iloc[:,:-1]
y=new_data['return']

from sklearn.model_selection import new_data_test_split
X_new_data,y_new_data,X_test,y_test=new_data_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_new_data=sc.fit_transform(X_new_data)
X_test=sc.transform(X_test)

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

model_GBM=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=10, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=50,
             min_samples_split=100, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)

pred_GBM=model_GBM.predict(new_data)

model_XGB=XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
       learning_rate=0.1, max_delta_step=0, max_depth=10,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='reg:linear', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)

pred_XGB=model_XGB.predict(new_data)


### Ensemble ####
final_pred=(0.6*pred_XGB+0.4*pred_GBM)

print("Done....\n")
"""














