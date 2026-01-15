import pandas as pd
import numpy as np
import shap 
import matplotlib.pyplot as plt
from tqdm import tqdm
#shap package to explain variable effect on model


from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

"""
Rolling-Window XGBoost Volatility Forecasting Model (with SHAP Explainability)

Main Purpose:
- Trains an XGBoost regression model using a rolling time window (default: 10 days)
- Forecasts next-day realized volatility
- Supports flexible inclusion of lagged features from multiple data sources:
  - Historical realized volatility (required)
  - Price levels
  - Reddit sentiment (VADER score)
  - Reddit posting/activity volume(number of daily comments)
  - News sentiment

Key Features:
- Strict rolling-window time-series cross-validation (prevents look-ahead bias)
- Out-of-sample prediction performance evaluation via RMSE
- SHAP-based feature importance and model interpretability for the most recent window

Data Requirements (daily_data.csv):
- Required columns: date, realized_vol(Volatility)
- Optional columns: price, Social media: vader_score, post_count, Report or news: news_sentiment
- Data must be sorted in ascending order by date
"""

#========================================
# Feature Engineering
#========================================

def add_lagged_features(df,columns,lags):
    """
    Add lagged features for selected columns.
    
    Input Parameters:
    df: pd.DataFrame (sorted by date ascending = True)
    columns: list (selected features)
    lags: list of lag periods
    
    Output:return pd.Dataframe
    """
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

#========================================
# Rolling XGBoost Volatility Model with SHAP
#========================================

class RollingXGBoostVolatility:
    """
    Rolling-window XGBoost model for volatility forecasting
    Support features selection and SHAP explainability
    """
    
    def __init__(self,
                 feature_cols,
                 target_col,
                 window_size = 10,
                 xgb_params = None):
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        
        #XGboost necessary parameters list
        #If no input parameters, use default xgb_params
        self.xgb_params = xgb_params or {
            "n_estimators":300,    #Use 300 estimating tree models
            "max_depth":3,         #max_depth for every tree is 3
            "learning_rate":0.05,  #new tree weight to estimate
            "subsample":0.8,
            "colsample_bytree":0.8, #80% features and 80% samples
            "random_state":42
        }
        #Containers to save results
        self.predictions_ = []
        self.actuals_ = []
        self.dates_ = [] #save estimate dates
        self.models_ = [] #save every XGBoost models
        
    def _build_model(self):
        """
        Initialize XGBoost model
        """
        return XGBRegressor(**self.xgb_params)
    
    def fit_predict(self,df):
        """
        Rolling-window training and prediction 
        
        Parameters: 
        df: pd.Dataframe Input dataframe sorted by date
        
        Returns:
        pd.DataFrame
            Prediction results.
        """
        
        if "date" not in df.columns:
            raise ValueError(
                "The input DataFrame must contain a column named 'date'. "
            "This column is required for sorting and recording prediction dates."
        )
        # date must be datetime type
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise TypeError(
            "The 'date' column must be of datetime type. "
            "Please convert it using pd.to_datetime() or parse_dates when reading the file."
        )
        # Date must 
        df = df.sort_values("date").reset_index(drop=True)
        
        for i in tqdm(range(self.window_size,len(df)),
                      desc = 'Training rolling windows',
                      unit = 'window'):
            train_df = df.iloc[i - self.window_size:i]
            test_df = df.iloc[i:i+1]
            
            X_train = train_df[self.feature_cols]
            y_train = train_df[self.target_col]
            
            X_test = test_df[self.feature_cols]
            y_test = test_df[self.target_col].values[0]
            
            model = self._build_model()
            model.fit(X_train,y_train)
            
            y_pred = model.predict(X_test)[0]
            
            self.predictions_.append(y_pred)
            self.actuals_.append(y_test)
            self.dates_.append(test_df['date'].values[0])
            self.models_.append(model)
        
        return self.results() #we use results function
    
    def results(self):
        """
        Return predictions results as a DataFrame
        """
        return pd.DataFrame({"date":self.dates_,
                             "predicted_volatility":self.predictions_,
                             "realized_volatility":self.actuals_})
    def rmse(self):
        """Compute RMSE of rolling predictions"""
        return np.sqrt(mean_squared_error(self.actuals_,self.predictions_))
    
    def shap_explain_last_window(self,df):
        """Compute SHAP values for the last trained rolling window.
        
        Returns
        explainer,shap_values,X_window
        """
        last_model = self.models_[-1]
        last_window = df.iloc[-self.window_size:][self.feature_cols]
        
        explainer = shap.TreeExplainer(last_model)
        shap_values = explainer.shap_values(last_window)
        
        return explainer,shap_values,last_window
    
def main(input_file_path):
    """
    Main execution script
    """
    #-------------------
    #Load daily data
    #Required columns:
    #date,price,realized_vol,vader_score,post_count,news_sentiment
    #-------------------
    df = pd.read_csv(input_file_path,parse_dates = ['date']) #read date as datetime
    
    #Select which variables to include
    use_price = True
    use_reddit_sentiment = True
    use_reddit_volume = True
    use_news_sentiment = False
    
    #Base_columns:Lag_columns;
    #if select several lagged variables to include:feature engineering
    lag_columns = ["realized_vol"]
    
    if use_price:
        lag_columns.append("price")
    if use_reddit_sentiment:
        lag_columns.append("vader_score")
    if use_reddit_volume:
        lag_columns.append("post_count")
    if use_news_sentiment:
        lag_columns.append("news_sentiment")
    #Here we have 5 different lagged feature
    
    #Feature Engineering
    df = add_lagged_features(df,lag_columns,lags=[1,3,7])
    df = df.dropna().reset_index(drop = True) #delete original index
    #This procedure: we add la
    
    #construct feature list dynamically
    feature_cols = []
    for col in lag_columns:
        for lag in [1,3,7]:#lag time period list
            feature_cols.append(f"{col}_lag{lag}")
    
    target_col = 'realized_vol'
    
    #------------------------------------
    #Train Rolling-XGBoost model
    #------------------------------------
    model = RollingXGBoostVolatility(feature_cols=feature_cols,
                                     target_col=target_col,
                                     window_size=10)
    
    results = model.fit_predict(df)
    #Here print Total RMSE of our rolling models
    print("Rolling RMSE:",model.rmse())
    
    
    #------------------------------------
    # Plot prediction results
    #------------------------------------
    plt.figure(figsize = (12,6))
    plt.plot(results["date"],results["realized_volatility"],label = "Realized Volatility")
    plt.plot(results["date"],results["predicted_volatility"],label = "Predicted Volatility")
    plt.legend()
    plt.title("Rolling XGBoost Volatility Forecast")
    plt.show()
    
    #------------------------------------
    # SHAP explainability (last rolling window)
    #------------------------------------
    explainer,shap_values,X_shap = model.shap_explain_last_window(df)
    shap.summary_plot(shap_values,X_shap)
    
if __name__ == '__main__':
    main(input_file_path)