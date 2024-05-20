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
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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
#3. EDA
#3.1. Check the distribution for "age" and "fare" features.
fig, ax = plt.subplots(figsize=(40,18))

plt.subplot(2,2,1)
sns.kdeplot(data=titanic_df,x='age')
plt.title('Age PDF')
plt.subplot(2,2,2)
stats.probplot(titanic_df['age'], dist="norm", plot=plt)
plt.title('Age QQ Plot')
plt.subplot(2,2,3)
sns.kdeplot(data=titanic_df,x='fare')
plt.title('Fare PDF')
plt.subplot(2,2,4)
stats.probplot(titanic_df['fare'], dist="norm", plot=plt)
plt.title('Fare QQ Plot')
plt.show()
#3.2. Check for skewness of pdf
print(f"Skew values:{titanic_df.skew(axis = 0)}")

#4.	Feature engineering and selection
#	Split data into training and test set
X_train,X_test,y_train,y_test = train_test_split(titanic_df.drop(columns=['survived']),
                                                 titanic_df['survived'],
                                                 test_size=0.2,random_state=42)
X_train.info()
#Pipeline with ML model
# Step 2.1 for each feature engineering task, create a transformer
#log transformer
log_trf = FunctionTransformer(func=np.log1p)
#################################################################################
#                    pipeline without the model(just log transformed)           #
#################################################################################
log_trans_pipe= make_pipeline(log_trf)
log_transformed_data=log_trans_pipe.fit_transform(X_train)
#check whether age and fare features were normalized or not
#3.1. Check the distribution for "age" and "fare" features.
fig, ax = plt.subplots(figsize=(40,18))

plt.subplot(2,2,1)
sns.kdeplot(data=log_transformed_data,x='age')
plt.title('Age PDF after log normalization')
plt.subplot(2,2,2)
stats.probplot(log_transformed_data['age'], dist="norm", plot=plt)
plt.title('Age QQ Plot after log normalization')
plt.subplot(2,2,3)
sns.kdeplot(data=log_transformed_data,x='fare')
plt.title('Fare PDF after log normalization')
plt.subplot(2,2,4)
stats.probplot(log_transformed_data['fare'], dist="norm", plot=plt)
plt.title('Fare QQ Plot after log normalization')
plt.show()

#We decided to only apply 1log transformation on the "fare" feature, not on the "age"
log_col_transformer = ColumnTransformer([
    ('log_trans_fare',FunctionTransformer(func=np.log1p),[1])
],remainder='passthrough')
#5. Model building
#create a transformer for model
logistic_reg_trf = LogisticRegression()
decision_tree_trf = DecisionTreeClassifier()
# Create a pipeline for all feature engineering steps (including ML model)
logistic_reg_pipe = make_pipeline(log_col_transformer,logistic_reg_trf)
decision_tree_pipe = make_pipeline(log_col_transformer,decision_tree_trf)
# #	5.1 Train the model using train data set
logistic_reg_pipe.fit(X_train,y_train)
decision_tree_pipe.fit(X_train,y_train)
#6. Model testing
#Test the model(s) built using the test data and evaluate the result.
#	6.1.run prediction
logistic_reg_prediction=logistic_reg_pipe.predict(X_test)
decision_tree_prediction=decision_tree_pipe.predict(X_test)
#	6.2 Evaluate the prediction
print(f"For logistic regression, accuracy score:{accuracy_score(y_test,logistic_reg_prediction)}")
print(f"For decision tree, accuracy score:{accuracy_score(y_test,decision_tree_prediction)}")