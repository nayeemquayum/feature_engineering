#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
#for pipeline
from sklearn.pipeline import Pipeline,make_pipeline
#for transformers
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


# To Display Pipeline
from sklearn import set_config

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#caling pipe.fit(taining_data) will produce a graphical representation of the pipeline
set_config(display='diagram')
#1. Gather data
titanic_df= sns.load_dataset('titanic')
#print(titanic_df.info())
#2. Data preprocessing
#2.1. Understand the data
# prof = ProfileReport(titanic_df)
# prof.to_file(output_file='output.html')
#2.2. Data Cleaning
#drop all the columns except survived, age and fare
titanic_df.drop(['pclass','sex','sibsp','parch','embarked',\
                       'class', 'who','adult_male','deck','embark_town',\
                       'alive','alone'], axis='columns', inplace=True)
#2.3. Make the dataframe more memory efficient
#down cast survived,pclass,sibsp,parch,fare column type
titanic_df['survived']=titanic_df['survived'].astype('int8')
titanic_df['fare']=titanic_df['fare'].astype('float32')
titanic_df['age']=titanic_df['age'].astype('float32')

#2.4. Handle missing values is age column.
#replace missing values from "age" column with the mean fare value
mean_age= titanic_df['age'].mean()
titanic_df.fillna({'age' : mean_age},inplace=True)
#2.5. Handle duplicate rows
#print(f"There are {titanic_df.duplicated().sum()} duplicated rows in the data.")
titanic_df.drop_duplicates(inplace=True)
# print("After dropping duplicated rows")
print(titanic_df.describe())
#Split data in training and testing set
X_train,X_test,y_train,y_test = train_test_split(titanic_df.drop(columns=['survived']),
                                                 titanic_df['survived'],test_size=0.2,random_state=42)
print(X_train.info())
#loop through all the columns to draw pdf and qqplot
for col in X_train.columns:
    figure = plt.figure('Before applying any Transformation',figsize=(14, 4))
    #figure.canvas.setWindowTitle('Before applying any Transformation')
    plt.subplot(1,2,1)
    sns.distplot(X_train[col])
    plt.title(col)

    plt.subplot(1,2,2)
    stats.probplot(X_train[col], dist="norm", plot=plt)
    plt.title(col)
    plt.show()
# Applying Box-Cox Transform

#apply box-cox power transformer
pt = PowerTransformer(method='box-cox')
pt_yeo_johnson = PowerTransformer()
X_train_transformed = pt_yeo_johnson.fit_transform(X_train)
X_test_transformed = pt_yeo_johnson.transform(X_test)
#tranformer functions returns numpy array. convert them to dataframe
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=['age', 'fare'])
box_cox_result=pd.DataFrame({'cols':X_train.columns,'yeo_johnson_lambdas':pt_yeo_johnson.lambdas_})
print(X_train_transformed_df.head())
print(box_cox_result.head())
print("After applying transformation the features PDF")
for col in X_train_transformed_df.columns:
    figure=plt.figure('After Yeo-Johnson Transformation',figsize=(14,4))
    plt.subplot(1,2,1)
    sns.distplot(X_train_transformed_df[col])
    plt.title(col)

    plt.subplot(1,2,2)
    stats.probplot(X_train_transformed_df[col], dist="norm", plot=plt)
    plt.title(col)
    plt.show()
# Applying linear regression on transformed data
lr = LinearRegression()
lr.fit(X_train_transformed,y_train)

y_lr_prediction = lr.predict(X_test_transformed)

print(f"r2 score for linear regression:{r2_score(y_test,y_lr_prediction)}")