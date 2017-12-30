## Import Libraries

import numpy as np
import pandas as pd
import sklearn


##Import Dataset

new_data=pd.read_csv("new_data.csv")

print("Preprocessing Started!! Please Wait...\n")

## Data Preprocessing ####

## Categorical Variables
cat_vars = [x for x in new_data.columns if 'cat_' in x]

## Missing Values
for x in cat_vars:
    new_data[x] = new_data[x].fillna(method='bfill')
    


### Label Encoding 

	###Used Numpy Union Method to capture all the labels present in dataset
labels=np.union1d(new_data.cat_var_3,new_data.cat_var_1)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
lb.fit(labels)

## Transforming all Categorical Variables(new_data Set)
for x in cat_vars:
    if new_data[x].dtype=='object':
        new_data[x]=lb.transform(new_data[x])



new_data.to_csv("Cleaned_new_data.csv",index=False)

    
print("Preprocessing Done!! Look for Cleaned_new_data.csv file...\n")


