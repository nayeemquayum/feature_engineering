#We'll do EDA, feature engineering and then run two ML algorithms on titanic data.
1. Using decision tree algorithm predict the survival chance of a passenger(pipeline.py).
2. Using both decision tree and logistic regression predict survival chance of a passenger 
only considering age and fare(pipeline2.py).
3. We try applying different transformer methods on features to find optimal way to convert
their PDF to normal distribution (power_transformer.py)
4. Despite using different scaling and transformer methods, we observed sub per accuracy rate for our ML models.
We'll try using various methods to convert numerical feature of "age" and "fare" to categorical feature and re run
or model to see whether we observe any improvement.
5. After observing the PDF of "age" and "fare category", decided to transform age to 8 uniform bins
and "fare" to 5 quantile bins. It produced the best accuracy for the decision 
tree(numerical_transformation_to_categorical.py).

We'll use 
-ydata-profiling to understanding the data
-EDA to check distribution of features (age and fare) and check whether they are normally distributed.
-pipeline and transformer to do feature engineering
-pickel to transport the model

Observation
1. We observed, the age column was missing 177 values. We replaced them with the mean
age of all the available samples.
2. Age feature had right skewed bimodal distribution with skewness 0.396337.  
3. Fare feature had very large right skewed distribution. 
4. After applying log distribution, we observed that "Age" feature transformed in to left
skewed distribution with many outliers.However, the PDF for "fare" got improved after log transformation. 
Therefore, we decided not to log transform the "age" feature, only 1log transformed the "fare"
feature(since there were 0 in the sample).