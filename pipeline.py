#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
#for pipeline
from sklearn.pipeline import Pipeline,make_pipeline
#for transformers
from sklearn.compose import ColumnTransformer
# To Display Pipeline
from sklearn import set_config
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#caling pipe.fit(taining_data) will produce a graphical representation of the pipeline
set_config(display='diagram')
#1. Gather data
titanic_df= sns.load_dataset('titanic')


#2. Data preprocessing
#2.1. Understand the data
prof = ProfileReport(titanic_df)
prof.to_file(output_file='output.html')

#2.2. Make the dataframe more memory efficient
#down cast survived,pclass,sibsp,parch,fare column type
titanic_df['survived']=titanic_df['survived'].astype('int8')
titanic_df['pclass']=titanic_df['pclass'].astype('int8')
titanic_df['sibsp']=titanic_df['sibsp'].astype('int8')
titanic_df['parch']=titanic_df['parch'].astype('int8')
titanic_df['fare']=titanic_df['fare'].astype('float32')
titanic_df['age']=titanic_df['age'].astype('float32')
#titanic_df.info()
#2.3. Data Integration: Resolving any conflicts in the data and handling redundancies.
#2.4. Data Cleaning
titanic_df.drop(['class', 'who','adult_male','deck','embark_town','alive','alone'], axis='columns', inplace=True)
#titanic_df.info()
#3. EDA

#4.	Feature engineering and selection
#split data in training and test set
X_train,X_test,y_train,y_test = train_test_split(titanic_df.drop(columns=['survived']),
                                                 titanic_df['survived'],
                                                 test_size=0.2,random_state=42)
#	4.1. Handle missing values
missing_values_imputer = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]),
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough',verbose_feature_names_out=False)
#   4.2. Feature scaling for numerical features
numerical_values_scaler = ColumnTransformer([
    ('scaled_age',MinMaxScaler(),[3]),
    ('scaled_sibsp',MinMaxScaler(),[5]),
    ('scaled_parch',MinMaxScaler(),[6]),
    ('scaled_fare',MinMaxScaler(),[7]),
],remainder='passthrough')
#	4.3. Encode input categorical features and label output variable (if any)
# one hot encoding
OHE_tranformer = ColumnTransformer([
    ('ohe_sex',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),[3]),
    ('ohe_embarked',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),[1])
],remainder='passthrough')

#	4.4. Outlier detection
#	4.5.  Feature Selection
FS_transformer = SelectKBest(score_func=chi2,k=4)
#	4.6.  Extract input and output columns
#	4.7. Train test split using cross validation method
#5. Model building
model_transformer = DecisionTreeClassifier()
#create the pipeline
pipe = make_pipeline(missing_values_imputer,OHE_tranformer,numerical_values_scaler,\
                     FS_transformer,model_transformer)
#	5.1 Train the model using train data set
temp=pipe.fit(X_train,y_train)

#6. Model testing
#Test the model(s) built using the test data and evaluate the result
#6.1.run prediction
y_prediction = pipe.predict(X_test)
# #6.2 Evaluate the prediction
print(f"accuracy score:{accuracy_score(y_test,y_prediction)}")
print(f"cross-validation score:{cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()}")
#make binary file of the pipe for export
pickle.dump(pipe,open('pipe.pkl','wb'))