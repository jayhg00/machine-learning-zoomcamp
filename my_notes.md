# My notes

- As part of EDA & data prep, check the correlation between pairs of each Input feature using **df.corr()**. Plot the same using seaborn heatmap using **sns.heatmap(df.corr(), annot=True)**.
- If any two input features show a high correlation (either +ve or -ve > 0.8 or 0.85), it is called **Multicollinearity** and this is a **problem**. It becomes difficult for the model to determine the individual effect of each correlated variable, leading to unreliable coefficient estimates. So, one way to solve is to **keep only one of the features as the predictor and discard the other correlated features**. So, in your prepare_X() method, ignore the correlated input features
