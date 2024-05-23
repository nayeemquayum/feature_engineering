#import packages
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

#1. Gather data
df=pd.read_csv("data/train.csv")
print(df.info())
#check for null values
print(df.isnull().mean()*100)
corr_df=df.corr()['price_range']
print(f"Correlation of price_range:{corr_df.sort_values(ascending=False)}")
#split data into train and test
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['price_range']),
                                                 df['price_range'],test_size=0.2,random_state=20)
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
best_features_score = bestfeatures.fit(X_train,y_train)
print(best_features_score.scores_)
scores_df = pd.DataFrame(best_features_score.scores_)
columns_df = pd.DataFrame(X_train.columns)
#concat two dataframes for better visualization
best_features_score_df = pd.concat([columns_df,scores_df],axis=1)
#naming the dataframe columns
best_features_score_df.columns = ['features','score']
print(best_features_score_df.sort_values('score',ascending=False))
#Calculate feature importance from ExtraTreesClassifier
#impor package
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X_train,y_train)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#create a series with feature's importance
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
#plot graph of feature importances for better visualization
feat_importances.nlargest(10).plot(kind='barh')
plt.show()