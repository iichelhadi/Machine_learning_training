#Imports
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv', index_col= ['id'])

# Encoding categorical data
labelencoder_y_1 = LabelEncoder()
df.diagnosis = labelencoder_y_1.fit_transform(df.diagnosis)

df = df.iloc[:,:-1]

X = add_constant(df)

l = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)

l = l.sort_values(ascending = False)

## cols to drop
cols = l[:11].index.drop('const')
## dropping columns with high VIF (multicolinear)
df1 = df.drop(cols, axis = 1)