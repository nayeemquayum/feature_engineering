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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
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
#run decission tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
print(f"accuracy for decision tree without any feature transformation {accuracy_score(y_test,y_predict)}")
#create Discretizer
age_transformer = KBinsDiscretizer(n_bins=8,encode='ordinal',strategy='uniform')
fare_transformer = KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='quantile')
#create the columnTransformer for binning
bin_transformer = ColumnTransformer([
    ('transformed_age',age_transformer,[0]),
    ('transformed_fare',fare_transformer,[1])
],remainder='passthrough')
#apply binning
X_train_transformed=bin_transformer.fit_transform(X_train)
X_test_transformed=bin_transformer.transform(X_test)
print(f"age category boundary:{bin_transformer.named_transformers_['transformed_age'].bin_edges_}")
print(f"fare category boundary:{bin_transformer.named_transformers_['transformed_fare'].bin_edges_}")

#Create a dataframe with original values and transformed categories
transformed_pd = pd.DataFrame({
    'age':X_train['age'],
    'transformed_age':X_train_transformed[:,0],
    'fare':X_train['fare'],
    'transformed_fare':X_train_transformed[:,1]
})

transformed_pd['age_cat_range'] = pd.cut(x=X_train['age'],
                                    bins=bin_transformer.named_transformers_['transformed_age']\
                                    .bin_edges_[0].tolist())
transformed_pd['fare_cat_range'] = pd.cut(x=X_train['fare'],
                                    bins=bin_transformer.named_transformers_['transformed_fare']\
                                    .bin_edges_[0].tolist())
print(transformed_pd.head())
#now let's run the decision tree with transformed features
dec_classifier = DecisionTreeClassifier()
dec_classifier.fit(X_train_transformed,y_train)
y_prediction_2 = dec_classifier.predict(X_test_transformed)

print(f"accuracy for decision tree after feature transformation {accuracy_score(y_test,y_prediction_2)}")
