# My notes

- As part of EDA & data prep, check the correlation between pairs of each Input feature using **df.corr()**. Plot the same using seaborn heatmap using **sns.heatmap(df.corr(), annot=True)**.
- If any two input features show a high correlation (either +ve or -ve > 0.8 or 0.85), it is called **Multicollinearity** and this is a **problem**. It becomes difficult for the model to determine the individual effect of each correlated variable, leading to unreliable coefficient estimates. So, one way to solve is to **keep only one of the features as the predictor and discard the other correlated features**. So, in your prepare_X() method, ignore the correlated input features

- **Performance metrics for Regression Models**
  - Mean squared error, MSE `from sklearn.metrics import mean_squared_error`. In this, due to squaring, the error's unit also get squared eg km^2, kg^2 etc
  - Mean absolute error, MAE `from sklearn.metrics import mean_absolute_error`. The error's unit is same as target variable eg km, kg
  - Root mean squared error, RMSE `from sklearn.metrics import root_mean_squared_error`. By taking root, the error's unit is same as target variable eg km, kg
  - R-squared `from sklearn.metrics import r2_score`. Lies between 0 & 1. Higher value towards 1 means better model
  - adjusted R-squared manual compute `1 - (1 - R²) * ((n - 1) / (n - p - 1)) where R²: is the standard R-squared value obtained from sklearn.metrics.r2_score.
n: is the number of observations (data points) in your dataset.
p: is the number of independent variables (features) in your model.`

- **Performance metrics for Classification Models**
  - Accuracy, (TP+TN)/(TP+FP+TN+FN) `from sklearn.metrics import accuracy_score`. Lies between 0 & 1
  - Precision, TP/(TP+FP) `from sklearn.metrics import precision_score`. Lies between 0 & 1
  - Recall, TP/(TP+FN) `from sklearn.metrics import recall_score`. Lies between 0 & 1
  - F1-score, `from sklearn.metrics import f1_score`. harmonic mean of precision and recall `2PR/(P+R)` . Lies between 0 & 1
  - AUC-ROC `from sklearn.metrics import roc_auc_score`. Lies between 0 & 1

# **ML Project Key Stages**
- Data Collection
- Exploratory Data analysis (EDA) to clean up data
    - Handle missing values,
    - wrong values,
    - outliers,
    - imbalanced dataset,
    - One-hot encoding
- Feature Engineering/Selection (Select a few of the top most **important** features as Predictors using Correlation. Remove Multicollinear features)
    - To quantify how importantly a **NUMERICAL feature** impacts the target, get the **CORRELATION SCORE between each numerical feature & target**. Higher the score, more important the feature
    - To quantify how importantly a **CATEGORICAL feature** impacts the target, get the **MUTUAL INFORMATION SCORE between each categorical feature & target**. Higher the score, more important the feature
- Train, validate different models with hyper parameter tunings & evaluate the metrics. Select the best model from this.
- Pickle the model & expose as API
- Deploy

### Handling missing values (NaN or blank) ###
- If the no of rows with missing values << total no of rows, then those rows can be dropped without losing much information using `df.dropna()`. Else, IMPUTE (compute & replace) the missing values
  - For NUMERICAL features, impute with Mean, Median Or Some Constant as per suitable use-case
  - For CATEGORICAL features, impute with Mode (Most Frequent) or Some Constant as per suitable use-case

```Python
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Sample data with missing values
data = pd.DataFrame({
    'numerical_feature': [1, 2, np.nan, 4, 5],
    'categorical_feature': ['A', 'B', 'A', np.nan, 'C']
})

# Impute numerical feature with the mean
imputer_numerical = SimpleImputer(strategy='mean')
data['numerical_feature'] = imputer_numerical.fit_transform(data[['numerical_feature']])

# Impute categorical feature with the most frequent value
imputer_categorical = SimpleImputer(strategy='most_frequent')
data['categorical_feature'] = imputer_categorical.fit_transform(data[['categorical_feature']])

print(data)
```

### Handling Imbalanced datasets ###
For a Classification problem, an Imbalanced dataset is one where one of the classes for Target is in majority than the other. Example, in a dataset for Churn Prediction with 1000 rows, the Target has 900 rows with "No Churn" and 100 rows with "Churn".  
Ways to solve -
1. Random UPSAMPLING the Minority class - creates new samples for the Minority class by duplicating existing samples. There will then be 900 rows for "No Churn" & 900 rows for "Churn". But, due to duplicates, it can cause Overfitting and so not preferred
2. Random DOWNSAMPLING the MAJORITY class - There will then be 100 rows for "No Churn" & 100 rows for "Churn" [NOT RECOMMENDED DUE TO INFORMATION LOSS]
3. **Synthetic Minority OverSampling Technique SMOTE** - constructs new samples by interpolating between existing minority-class samples. There will then be 900 rows for "No Churn" & 900 rows for "Churn"
  ```Python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
  ```
5. Evaluate models using metrics that account for imbalance like **AUC Score**
6. **Use ensemble models like Random Forest or Boosting which inherently handle imbalances**

### Data encoding Categorical features ###
Encoding means assigning numerical value to Categorical value
#### One hot encoding ####
- Create new feature column for each value of Categorical feature and put "1" in respective columns and 0 in other columns. For color=RED|BLUE|GREEN, 3 new columns color_RED, color_BLUE, color_GREEN created and populated with 1 and 0
- If there are lot of possible values, then that many new columns will get created. So, avoid in such scenario

#### Label encoding Or Ordinal Encoding ####
- Simply replace the Categorical value with constant numerical value. RED=1, BLUE=2, GREEN=3.
- Here, the model may incorrectly treat GREEN to be more important than RED. So, use this for category where the numbers can reperesent the importance like Degree = HighSchool|Bachelors|Masters|PHD where HighSchool = 1, Bachelors = 2, Masters = 3, PHD = 4

## ML Steps ##
1. Split the dataset into Training & Test using `train_test_split()` to get X_train, X_test, y_train, y_test. Keep aside the X_test, y_test as we don't want the model to know anything about the test data
2. Perform EDA on X_train, y_train by using `.fit_transform(X_train)` methods of applicable operations. Eg: Imputation, One-hot Encoding etc
3. Apply the EDA operations on the X_test, y_test using `.transform(X_test)` methods of applicable operations.
4. Train the model on X_train, y_train using `model.fit(X_train, y_train)`
5. Make predictions on X_test using `y_pred=model.predict(X_test)`
6. Evaluate metrics of the model on y_pred, y_test using appropriate scores (accuracy, auc etc)


### Ridge Regression (L2 Regularization) ###
**To reduce overfitting**
When to choose ridge regression 
- When multicollinearity is present: Ridge regression is designed to handle situations where independent variables are highly correlated, which can make ordinary least squares (OLS) regression unstable and produce large, unreliable coefficients.
- To reduce overfitting: It adds a penalty term (lambda * coeff^2) that shrinks coefficients toward zero, preventing them from becoming too large and making the model less sensitive to noise in the training data.
- When all predictors are important: If you expect most of your features to have some predictive power, ridge regression is a good choice because it keeps all predictors in the model, albeit with reduced influence.
- When you have more predictors than observations: Ridge regression can provide a stable solution in cases where \(n<p\) (where \(n\) is the number of observations and \(p\) is the number of predictors), unlike OLS which may not have a unique solution.

### Lasso (Least Absolute Shrinkage and Selection Operator) Regression L1 Regularization ###
**To perform feature selection**
- Adds penalty term (lamdba * abs(coeff)) that forces some coefficients to become exactly zero meaning the features with zero coeff do not have any role in the target variable and such features can be dropped from the model ==> Enables feature selection

Use Lasso when
- When you have a high number of features, and you suspect many of them are not important for the prediction.
- To automatically select the most relevant features from a large dataset.
- When dealing with multicollinearity (high correlation between predictor variables).

### ElasticNet Regression (Lasso L1 + Ridge L2) ###
- Combines benefits of Lasso & Ridge to do feature selection & to reduce overfitting.
- Adds penalty terms of both Lasso (lambda * abs(coeff)) + Ridge (lambda * coeff^2)
- In sklearn's ElasticNet,
    - "alpha" controls overall regularization strength (higher alpha --> higher shrinkage).
    - "l1-ratio" Controls the balance between the L1 and L2 penalties
        - 0 => Ridge regression
        - 1 => Lasso
        - 0<val<1 ==> Hybrid

When to use
- High-dimensional data: When the number of features is greater than the number of samples. 
- Correlated features: When you have a group of highly correlated predictors. (high multi-collinearity)
- Feature selection and regularization: When you want to perform both feature selection and regularize the model's coefficients simultaneously. 

## Hyperparameter Tuning ##
Hyperparameters are the parameters that are set before the machine learning model training process begins. You set these hyperparameters when instantiating the model variable or after instantiation

`sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)`
For Logistic Regression, all the arguments are hyperparameters

Tuning means to determine the combination of parameter values which will give best score of the model for a given dataset.

### GridSearchCV ###
Defines a "grid" of hyperparameter values. For each hyperparameter that you want to tune, you specify a list of potential values to test. GridSearchCV then exhaustively tries every possible combination of these specified values.
Process-
- You define a machine learning model (estimator).
- You create a dictionary (param_grid) where keys are hyperparameter names and values are lists of values to test for each hyperparameter.
- You initialize GridSearchCV with the estimator, param_grid, and a scoring metric (e.g., accuracy, F1-score).
- You fit the GridSearchCV object to your training data.
- After fitting, GridSearchCV identifies the hyperparameter combination that yielded the best performance based on the chosen scoring metric and cross-validation results.
- Output: GridSearchCV.best_params_ (the optimal hyperparameter combination) and best_score_ (the best performance achieved) found during the search. You can then use these optimal parameters to train your final model on the entire training dataset.

```Python
# Set the model and the hyperparameters of interest to tune
model=LogisticRegression()
penalty=['l1', 'l2', 'elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Create Params dictionary
params=dict(penalty=penalty,C=c_values,solver=solver)

# Tune using GridSearchCV
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=model,param_grid=params,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)

grid.best_params_ ## {'C': 0.01, 'penalty': 'l1', 'solver': 'saga'}
grid.best_score_ ## 0.9242857142857142

# Predict using the tuned hyperparameters
y_pred=grid.predict(X_test)
score=accuracy_score(y_pred,y_test) ## 0.92
```
### RandomizedSearchCV ###
Instead of trying every single possible combination like GridSearchCV, RandomizedSearchCV randomly selects a specified number of parameter combinations. Useful when hyperparameter search space is large and an exhaustive search is computationally infeasible or time-consuming.
```Python
# Set the model and the hyperparameters of interest to tune
model=LogisticRegression()
penalty=['l1', 'l2', 'elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Create Params dictionary
params=dict(penalty=penalty,C=c_values,solver=solver)

# tune using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,cv=5,scoring='accuracy')
randomcv.fit(X_train,y_train)

randomcv.best_score_ ## 0.9128571428571428
randomcv.best_params_ ## {'solver': 'saga', 'penalty': 'l2', 'C': 100}

# Predict using the tuned hyperparameters
y_pred=randomcv.predict(X_test)
score=accuracy_score(y_pred,y_test) ## 0.9166
```

## Support Vector Machines ##
- Used for both Regression (Support Vector Regression SVR) & Classification (Support Vector Classification) tasks.
- Useful to find best fit curve/decision boundary in overlapping/linearly non-separable data points (Linear Regression or Logistic Regression will not give good results)
- **Concept**: Does this by transforming the linearly non-separable data points from n-dimension space to (n+1)-dimension space where the points become linearly separable. In (n+1)-dimension space, there is clear separation of data points and so best fit line/curve is found which is re-transfomed to n-dimension space.
  - Example - points on a 1-dimension number line (x) is transformed to a 2-dimension plot (x-y) using function y=x^2. Best fit line/curve is found in the 2-dimension space and re-tranposed on the 1-dimension number line.
  - Example - points on a 2-dimension space (x-y) is transformed to a 3-dimension space (x-y-z) using some functions. Best fit line/curve is found in the 3-dimension space and re-tranposed on the 2-dimension space.
  - **Kernel Trick**
    - This concept of transforming data points from n-dimension to (n+1)-dimension and back is computationally intensive and time-consuming. So, in practical cases, instead of explicitly performing the transformation for every data point, a **kernel function** is used. This function takes two data points as input and directly calculates their dot product in the higher-dimensional space without ever having to compute the coordinates of the points in that new space. Avoids the computational cost of transforming the entire dataset into a very high or even infinite-dimensional space, making the process much more efficient. 
  - Types of Kernel function
    - RBF (Radial Basis Function) ==> Most widely used
    - Sigmoid
    - Polynomial

## K-NN K-Nearest Neighbours ##
- Simplest of all algorithms. Used for both classfication & regression.
- Concept : similar data points are located close to each other in a feature space and can be classified based on the proximity of their neighbors.
- K = 5 (default)
- For classification, plot existing data points. For any new data point, identify the K-nearest neighbours using distance formula. Out of those K-nearest neighbours, assign the majority class by count.
- For regression, predict the value of the new data point by calculating the average of the values of the k neighbors.
- Types of distance-
  - Euclidian distance = Direct distance between two points. ==> sqrt((x2-x1)^2 + (y2-y1)^2)
  - Manhattan distance = Distance traversed as though in Manhattan city blocks ==> abs(x2-x1) + abs(y2-y1)
- Optimizing the algorithm:
  - In a large dataset, it is time-consuming to calculate distances for every data point for a new data point to identify the k-neighbours. So, to reduce the time by half, **KD-Tree** or **Ball Tree** algorithm is applied 

# Unsupervised Learning #
- **No EXPLICIT output (dependent) variable, y. All columns in dataset are the independent features X.**
- **There is no train test split. The whole dataset is used to fit and predict the clusters. The predictions will be the output(dependent) variable** 
- Used for identifying clusters/patterns which otherwise are not easily identified
- Example - Customer segmentation, image compression, pattern recognition
- Algorithms
  - k-Means Clustering
  - Hierarchical Clustering
  - DBSCan Clustering
  - Silhoutte Clustering

**NOTE: In all these algorithms, even when the input X has multiple features, we select any two important features only so that you can visualize the scatter plots & clusters for understanding. In real world apps, you will use all the important features (>2) and not necessarily visualize**

## k-Means Clustering #
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/b08ea8ee-053d-4ac9-96e6-54540e6cd897" />

**WCSS** : Within Cluster Sum of Squared distance (between data points in the cluster and the cluster's centroid)

**Elbow Method**: 
- Plot WCSS for different value of k. As k increases from 1 to number of datapoints, WCSS reduces towards 0.
- For k = number of datapoints (i.e. each cluster has only one data point and the data point itself is the centroid) ==> WCSS = 0.
- Starting with k=1, WCSS reduces quickly and after a certain k, it reduces slowly. This inflection point determines the "k" that should be set for the final model

**k-Means++ to avoid Random Initialization trap**
- If the initial centroids are randomly set whenever the K-means is run, then the clustering output will not be consistent across different runs ==> **Random Initialization Trap**
- To avoid this, k-Means++ algorithm is applied to set the initial centroids
  1. First centroid: A data point is selected at random from the dataset to be the first centroid.
  2. Subsequent centroids: For each remaining data point, the algorithm calculates the distance to the nearest centroid that has already been selected.
  3. Weighted selection: The next centroid is chosen randomly from the remaining data points, with the probability of selection being proportional to its squared distance to the nearest existing centroid. This ensures that points that are far from the initial centroids are more likely to be selected as new centroids, which helps spread them out.
  4. Repeat: Steps 2 and 3 are repeated until the desired number of centroids (\(k\)) has been selected.
  5. Standard k-means: Once the initial centroids are chosen by the k-means++ method, the standard k-means clustering algorithm is applied to the data to form the final clusters.


## Hierarchical Clustering ##
Unsupervised machine learning algorithm that groups data into a tree of nested clusters
Two types - 
- Agglomerative (bottom to top approach)
- Divisive (top to bottom approach)
Visualized using a DENDROGRAM tree-like structure. x-axis of DENDROGRAM are the individual datapoints while y-axis is the distance (dissimilarity).


### Agglomerative Hierarchical Clustering ###
- bottom-up approach that starts with each data point as its own cluster and then merges the two closest clusters in a series of steps until only one cluster remains

<img border="1" width="1840" height="730" alt="image" src="https://github.com/user-attachments/assets/dfcfdcc4-fce5-499f-8057-eef5af863715" />
In above image, P2 & P3 are closest so group those two with two vertical lines whose height is equal to the Euclidean distance between them (0.5)
<br/><br/>

<img border="1" width="1838" height="714" alt="image" src="https://github.com/user-attachments/assets/56c4fd71-9dc1-4ae9-9f56-c9354449bd78" />
Then P5,P6 are closest so group them with height = Euclidean distance between them (0.8)
<br/><br/>

<img border="1" width="1844" height="705" alt="image" src="https://github.com/user-attachments/assets/71ef79b5-9741-4641-8d3e-1647a5b32ece" />
Then P1, P2-P3 cluster are closer so group them with height = Euclidean distance between them (0.85)
<br/><br/>

<img border="1" width="1814" height="719" alt="image" src="https://github.com/user-attachments/assets/637da346-2b9b-4125-8c87-c1ce01437736" />
Then, P4, P5-P6 cluster are closer so group them with height = Euclidean distance between them (0.85)
<br/><br/>

<img border="1" width="1825" height="723" alt="image" src="https://github.com/user-attachments/assets/20629ee1-8d25-4536-b4ca-129538c7bd28" />
Then, P1-P2-P3, P4-5-6 cluster are grouped to form one single cluster with height = Euclidean distance between them (2.7)
<br/><br/>

**Then to know the optimal number of clusters,**
<img width="1805" height="676" alt="image" src="https://github.com/user-attachments/assets/f216c160-4f3f-42b0-bb32-ed66e9bff1ba" />
- Find the largest separation between the horizontal lines and draw a horizontal line at its middle. In above image, largest separation is as shown
- Where this horizontal line cuts the vertical lines, those vertical lines are the final clusters. In above image, two vertical lines are cut and the clusters are P1-2-3 & P4-5-6


