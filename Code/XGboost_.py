import pandas as pd
import numpy as np
import shap 
import matplotlib.pyplot as plt
from tqdm import tqdm
#shap package to explain variable effect on model


from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
                 window_size = 50,
                 xgb_params = None):
        
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        
        #XGboost necessary parameters list
        #If no input parameters, use default xgb_params
        self.xgb_params = xgb_params or {
            "n_estimators":800,    #Use 300 estimating tree models
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
    
    def rmse_original(self):
        if len(self.actuals_) == 0:
             return None
        actual = np.array(self.actuals_)
        pred = np.array(self.predictions_)
        actual_original = np.expm1(actual)
        pred_original = np.expm1(pred)
        return np.sqrt(mean_squared_error(actual_original, pred_original))
    
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
    
    def r2(self):
        """
        Compute R-squared (coefficient of determination) of rolling predictions.
        R² = 1 - (SS_res / SS_tot), where higher is better (max 1.0)
        Returns None if no predictions were made.
        """
        if len(self.actuals_) == 0 or len(self.predictions_) == 0:
            return None
        return r2_score(self.actuals_, self.predictions_)
    
    def r2_original(self):
        if len(self.actuals_) == 0:return None
        actual = np.expm1(np.array(self.actuals_))
        pred = np.expm1(np.array(self.predictions_))
        return r2_score(actual, pred)
    
def main(input_file_path,window_size = 3,xgb_params = {
                    "n_estimators": 800,
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "objective": "reg:squarederror"
                },lags=[1,3,7],
         if_use_log_return = True,
         use_realized_volatility = True,
         use_price = False,
         use_reddit_sentiment = True,
         use_reddit_volume = False,
         use_news_sentiment = False):
    """
    Main execution script for the Rolling XGBoost Volatility Forecasting Model.
    
    This function:
    1. Loads and prepares the data
    2. Performs feature engineering with lagged features
    3. Trains the rolling XGBoost model
    4. Prints overall RMSE
    5. Plots predictions
    6. Shows SHAP explainability
    7. Performs hyperparameter tuning to find the best window_size, max_depth, and n_estimators
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input CSV file
    """
    # Load daily data
    # Required columns: date, realized_vol
    # Optional: price, vader_score, post_count, news_sentiment
    df = pd.read_csv(input_file_path, parse_dates=['date'])
    
    # Print data info for debugging
    print("Loaded data columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())

    
    # Base columns always included (historical volatility is required)
    if if_use_log_return:
        print("Using log-transformed target(realized_vol_log)")
        df['realized_vol'] = np.log1p(df['realized_vol'])
        if use_realized_volatility:
            lag_columns = ["realized_vol"]
        else:
            lag_columns = []
        target_col = 'realized_vol'
    else:
        print("Using original target(no log)")
        if use_realized_volatility:
            lag_columns = ["realized_vol"]
        else:
            lag_columns = []
        target_col = 'realized_vol'
        
    if use_price:
        lag_columns.append("price")
    if use_reddit_sentiment:
        lag_columns.append("vader_score")
    if use_reddit_volume:
        lag_columns.append("post_count")
    if use_news_sentiment:
        lag_columns.append("news_sentiment")
    
    # Feature Engineering: add lagged features
    df = add_lagged_features(df, lag_columns, lags)
    
    if if_use_log_return:
        df['realized_vol_original'] = np.expm1(df['realized_vol'])
    
    df = df.dropna().reset_index(drop=True)  # Drop NaNs and reset index
    
    print(df.head(3))
    print('Samples Start:{}'.format(df['date'].min()))
    print('Samples End:{}'.format(df['date'].max()))
    print(f'Window_size:{window_size}')
    print(f'If use log volatility:{if_use_log_return}')
    
    
    # Construct feature list dynamically
    feature_cols = []
    for col in lag_columns:
        for lag in lags:
            feature_cols.append(f"{col}_lag{lag}")
    
    
    print(f'feature_cols:{feature_cols}')
    print(f'xgb_params:{xgb_params}')
    
    # Train initial model with default parameters
    model = RollingXGBoostVolatility(
        feature_cols=feature_cols,
        target_col=target_col,
        window_size=window_size,
        xgb_params=xgb_params
    )
    
    results = model.fit_predict(df)
    #Log R-square and RMSE
    rmse_log = model.rmse()
    r2_log = model.r2()
    
    #Original R-square and RMSE
    rmse_original = model.rmse_original()
    r2_original = model.r2_original()
        
    if if_use_log_return:
        print(f"Rolling RMSE (log scale): {rmse_log:.6f}")
        print(f"R-squared (log scale): {r2_log:.6f}" if r2_log is not None else "R-squared (log scale): N/A")
        print(f"Rolling RMSE (original scale): {rmse_original:.6f}")
        print(f"R-squared (original scale): {r2_original:.6f}" if r2_original is not None else "R-squared (original scale): N/A")
        results['predicted_volatility'] = np.expm1(results['predicted_volatility'])
        print("Predicted values have been transformed back to original scale using expm1")
    else:
        print(f"Rolling RMSE: {rmse_log:.6f}  (original)")
        print(f"R-squared (R²): {r2_log:.6f}" if r2_log is not None else "R-squared: N/A")
    
    # Plot prediction results
    plt.figure(figsize=(14, 7))
    if if_use_log_return:
        plt.plot(results["date"], np.expm1(results["realized_volatility"]), label="Realized Volatility", linewidth=2)
        plt.plot(results["date"], results["predicted_volatility"], label="Predicted Volatility", linewidth=2, linestyle='--')
    else:
        plt.plot(results["date"], results["realized_volatility"], label="Realized Volatility", linewidth=2)
        plt.plot(results["date"], results["predicted_volatility"], label="Predicted Volatility", linewidth=2, linestyle='--')

    plt.legend(fontsize=12)
    plt.title("Rolling XGBoost Volatility Forecast", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Volatility", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # SHAP explainability for the last rolling window
    explainer, shap_values, X_shap = model.shap_explain_last_window(df)
    log_str = "log" if if_use_log_return else "nolog"
    lag_str = "".join(map(str, lags)) if isinstance(lags, list) else str(lags)
    exp_id = f"w{window_size}_{log_str}_lag{lag_str}_d{xgb_params['max_depth']}_t{xgb_params['n_estimators']}"

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title(f"SHAP Summary - {exp_id}", fontsize=14)
    plt.tight_layout() 
    plt.show() 

class StaticXGBoostVolatility:
    """
    Static split XGBoost model for volatility forecasting.
    Trains once on the first X% of data and predicts on the remaining (1-X)%."""
    def __init__(self, feature_cols, target_col, train_split=0.8, xgb_params=None):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.train_split = train_split
        self.xgb_params = xgb_params or {
            "n_estimators": 800,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 42
        }
        self.model = XGBRegressor(**self.xgb_params)
        self.predictions_ = None
        self.actuals_ = None
        self.dates_ = None

    def fit_predict(self, df):
        # order by date
        df = df.sort_values("date").reset_index(drop=True)
        
        # 
        split_idx = int(len(df) * self.train_split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train, y_train = train_df[self.feature_cols], train_df[self.target_col]
        X_test, y_test = test_df[self.feature_cols], test_df[self.target_col]

        # train 
        print(f"Training on {len(train_df)} samples, Testing on {len(test_df)} samples...")
        self.model.fit(X_train, y_train)

        # predict test
        self.predictions_ = self.model.predict(X_test)
        self.actuals_ = y_test.values
        self.dates_ = test_df['date'].values

        return pd.DataFrame({
            "date": self.dates_,
            "predicted_volatility": self.predictions_,
            "realized_volatility": self.actuals_
        })

    def rmse(self):
        return np.sqrt(mean_squared_error(self.actuals_, self.predictions_))

    def r2(self):
        return r2_score(self.actuals_, self.predictions_)
    
    def shap_explain(self, X_test):
        """SHAP Explainer"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        return explainer, shap_values
    

def main_static(input_file_path, train_split=0.8, lags=[1,3,7], 
                if_use_log_return=True, use_reddit_sentiment=True):
    """
    Static Train-Test Split version of the XGBoost Volatility model.
    """
    df = pd.read_csv(input_file_path, parse_dates=['date'])
    
    # feature enginerring
    target_col = 'realized_vol'
    lag_columns = ["realized_vol"]
    if if_use_log_return:
        df[target_col] = np.log1p(df[target_col])
    
    if use_reddit_sentiment:
        lag_columns.append("vader_score")
        
    df = add_lagged_features(df, lag_columns, lags)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [f"{col}_lag{l}" for col in lag_columns for l in lags]

    # model_fit
    model = StaticXGBoostVolatility(
        feature_cols=feature_cols,
        target_col=target_col,
        train_split=train_split
    )

    results = model.fit_predict(df)
    
    print(f"Static Split RMSE (log scale): {model.rmse():.6f}")
    print(f"Static Split R2: {model.r2():.6f}")

    if if_use_log_return:
        results['predicted_vol_orig'] = np.expm1(results['predicted_volatility'])
        results['realized_vol_orig'] = np.expm1(results['realized_volatility'])
    else:
        results['predicted_vol_orig'] = results['predicted_volatility']
        results['realized_vol_orig'] = results['realized_volatility']

    plt.figure(figsize=(12, 6))
    plt.plot(results['date'], results['realized_vol_orig'], label='Actual')
    plt.plot(results['date'], results['predicted_vol_orig'], label='Predicted', alpha=0.7)
    plt.title(f"Static Split Prediction (Train {train_split*100}%)")
    plt.legend()
    plt.show()

    split_idx = int(len(df) * train_split)
    X_test = df.iloc[split_idx:][feature_cols]
    explainer, shap_values = model.shap_explain(X_test)
    shap.summary_plot(shap_values, X_test)
    
    
if __name__ == '__main__':
    main()