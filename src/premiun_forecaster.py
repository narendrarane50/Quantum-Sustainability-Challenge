import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import warnings

# Suppress harmless warnings for clean console output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PremiumForecaster:
    """
    Panel Time-Series forecasting pipeline for Insurance Premiums.
    Leverages historical lags and hybrid Quantum/Classical fire risk scores.
    """
    def __init__(self, historical_data_path: str, task1_preds_path: str):
        self.raw_data_path = historical_data_path
        self.task1_preds_path = task1_preds_path
        
    def load_and_merge(self) -> pd.DataFrame:
        logging.info("Loading main dataset and Task 1 Quantum predictions...")
        # 1. Load main dataset
        df = pd.read_csv(self.raw_data_path)
        
        # 2. Load Task 1 predictions and rename keys for merging
        t1_preds = pd.read_csv(self.task1_preds_path)
        t1_preds = t1_preds.rename(columns={
            'zip_code': 'ZIP', 
            'predicted_fire_2023': 'quantum_fire_risk'
        })
        
        # 3. Merge Quantum features into historical data
        df = df.merge(t1_preds[['ZIP', 'quantum_fire_risk']], on='ZIP', how='left')
        
        # Assume 0 risk if the Quantum model didn't cover a specific ZIP
        df['quantum_fire_risk'] = df['quantum_fire_risk'].fillna(0)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Engineering autoregressive time-lags...")
        # Sort chronologically to safely apply historical shifts
        df = df.sort_values(by=['ZIP', 'Year']).reset_index(drop=True)
        
        # Actuarial Best Practice: Predict 'Premium Rate' (Premium per Exposure) 
        # instead of raw Premium to account for population changes.
        df['Premium_Rate'] = df['Earned Premium'] / (df['Earned Exposure'] + 1e-5)
        
        # Create Lagged Features (What happened in time t-1)
        df['Premium_Rate_Lag1'] = df.groupby('ZIP')['Premium_Rate'].shift(1)
        df['Avg_Fire_Risk_Score_Lag1'] = df.groupby('ZIP')['Avg Fire Risk Score'].shift(1)
        
        # Drop 2018 since we don't have 2017 data to act as its Lag1
        df = df.dropna(subset=['Premium_Rate_Lag1'])
        return df

    def run_pipeline(self):
        df = self.load_and_merge()
        df = self.engineer_features(df)
        
        # Define competitive feature sets
        base_features = ['Premium_Rate_Lag1', 'total_population', 'median_income', 'year_structure_built']
        
        models_to_test = {
            "Default Risk Only": base_features + ['Avg_Fire_Risk_Score_Lag1'],
            "Quantum Risk Only": base_features + ['quantum_fire_risk'],
            "Hybrid Synergy (Combined)": base_features + ['Avg_Fire_Risk_Score_Lag1', 'quantum_fire_risk']
        }
        
        # Proper Time-Series Splitting
        train_df = df[df['Year'] == 2019]  # Train on 2019 (using 2018 lags)
        val_df = df[df['Year'] == 2020]    # Validate on 2020 (using 2019 lags)
        test_df = df[df['Year'] == 2021]   # Predict target year 2021
        
        target = 'Premium_Rate'
        best_model = None
        best_rmse = float('inf')
        
        logging.info("--- Initiating A/B/C Model Evaluation ---")
        
        for name, features in models_to_test.items():
            # Robust imputation for missing census data using medians
            X_train = train_df[features].fillna(train_df[features].median())
            y_train = train_df[target]
            
            X_val = val_df[features].fillna(train_df[features].median())
            y_val = val_df[target]
            
            # Initialize Gradient Boosted Tree
            model = xgb.XGBRegressor(
                n_estimators=150, 
                learning_rate=0.05, 
                max_depth=5, 
                random_state=42,
                objective='reg:squarederror'
            )
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            logging.info(f"{name.ljust(25)} | Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = (name, model, features)
                
        logging.info(f"WINNING ARCHITECTURE: {best_model[0]}")
        
        # Generate official 2021 Predictions
        logging.info("Generating final 2021 predictions...")
        X_test = test_df[best_model[2]].fillna(train_df[best_model[2]].median())
        final_rate_preds = best_model[1].predict(X_test)
        
        # Convert rate back to standard Earned Premium
        results_2021 = test_df[['ZIP', 'Year', 'Earned Exposure']].copy()
        results_2021['Predicted_Premium_Rate'] = final_rate_preds
        results_2021['Predicted_Earned_Premium'] = final_rate_preds * results_2021['Earned Exposure']
        
        # Save output
        output_path = "Quantum-Sustainability-Challenge/results/task2_2021_premium_predictions.csv"
        results_2021[['ZIP', 'Predicted_Earned_Premium']].to_csv(output_path, index=False)
        logging.info(f"Mission Accomplished. Pipeline saved results to: {output_path}")

if __name__ == "__main__":
    # Update paths to match your exact directory structure
    forecaster = PremiumForecaster(
        historical_data_path="Quantum-Sustainability-Challenge/data/raw/cal_insurance_fire_census_weather.csv",
        task1_preds_path="Quantum-Sustainability-Challenge/results/wildfire_predictions_2023.csv"
    )
    forecaster.run_pipeline()