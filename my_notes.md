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
  - Accuracy , `from sklearn.metrics import accuracy_score`. Lies between 0 & 1
  - Precision, `from sklearn.metrics import precision_score`. Lies between 0 & 1
  - Recall, `from sklearn.metrics import recall_score`. Lies between 0 & 1
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

