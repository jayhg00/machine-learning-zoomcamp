# My notes

- As part of EDA & data prep, check the correlation between pairs of each Input feature using **df.corr()**. Plot the same using seaborn heatmap using **sns.heatmap(df.corr(), annot=True)**.
- If any two input features show a high correlation (either +ve or -ve > 0.8 or 0.85), it is called **Multicollinearity** and this is a **problem**. It becomes difficult for the model to determine the individual effect of each correlated variable, leading to unreliable coefficient estimates. So, one way to solve is to **keep only one of the features as the predictor and discard the other correlated features**. So, in your prepare_X() method, ignore the correlated input features

- **Performance metrics for Regression Models**
  - Mean squared error, MSE `from sklearn.metrics import mean_squared_error`
  - Mean absolute error, MAE `from sklearn.metrics import mean_absolute_error`
  - Root mean squared error, RMSE `from sklearn.metrics import root_mean_squared_error`
  - R-squared `from sklearn.metrics import r2_score`
  - adjusted R-squared manual compute `1 - (1 - R²) * ((n - 1) / (n - p - 1)) where R²: is the standard R-squared value obtained from sklearn.metrics.r2_score.
n: is the number of observations (data points) in your dataset.
p: is the number of independent variables (features) in your model.`

- **Performance metrics for Classification Models**
  - Accuracy , `from sklearn.metrics import accuracy_score`
  - Precision, `from sklearn.metrics import precision_score`
  - Recall, `from sklearn.metrics import recall_score`
  - F1-score, `from sklearn.metrics import f1_score`
  - AUC-ROC `from sklearn.metrics import roc_auc_score`
