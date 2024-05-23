#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
#for encoding
from sklearn.preprocessing import OneHotEncoder
#for pipeline
from sklearn.pipeline import Pipeline,make_pipeline
#for transformers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# To Display Pipeline
from sklearn import set_config

import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#caling pipe.fit(taining_data) will produce a graphical representation of the pipeline
set_config(display='diagram')
#1. Gather data
titanic_df= sns.load_dataset('titanic')
#print(titanic_df.info())
#2. Data preprocessing
#2.1. Understand the data
prof = ProfileReport(titanic_df)
# prof.to_file(output_file='output.html')
#2.2. Data Cleaning
#drop all the columns except survived, age and fare
titanic_df.drop(['pclass','sibsp','parch',\
                       'class', 'who','adult_male','deck','embark_town',\
                       'alive','alone'], axis='columns', inplace=True)
#2.3. Make the dataframe more memory efficient
#down cast survived,pclass,sibsp,parch,fare column type
titanic_df['survived']=titanic_df['survived'].astype('int8')
titanic_df['fare']=titanic_df['fare'].astype('float32')
titanic_df['age']=titanic_df['age'].astype('float32')
#Check for each columns, what percentage of values are missing
#print(titanic_df.isnull().mean()*100)
#split data into train and test
#Split data in training and testing set
X_train,X_test,y_train,y_test = train_test_split(titanic_df.drop(columns=['survived']),
                                                 titanic_df['survived'],test_size=0.2,random_state=42)
print("X_train data:")
print(X_train.head(2))
#create a pipeline for handling numerical data
numerical_features = ['age', 'fare']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
#create another pipeline for handling categorical data
categorical_features = ['sex','embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',OneHotEncoder(handle_unknown='ignore',drop='first'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
log_reg_classifier=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
log_reg_classifier.fit(X_train,y_train)
y_prediction=log_reg_classifier.predict(X_test)
# set_config(display='diagram')
# param_grid = {
#     'preprocessor__num__imputer__strategy': ['mean', 'median'],
#     'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant'],
#     'classifier__C': [0.1, 1.0, 10, 100]
# }
#
# grid_search = GridSearchCV(log_reg_classifier, param_grid, cv=10)
# grid_search.fit(X_train, y_train)
# y_prediction=grid_search.predict(X_test)
from sklearn.metrics import accuracy_score
print(f"accuracy score:{accuracy_score(y_test,y_prediction)}")

# print(f"Best params:")
# print(grid_search.best_params_)