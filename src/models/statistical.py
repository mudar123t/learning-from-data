import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_arima_model(data, order=(1, 1, 1), target_column='value'):
    """
    Train an ARIMA model on the given time series data.
    
    Args:
        data (pd.DataFrame): Time series data with a target column.
        order (tuple): The (p, d, q) order of the ARIMA model.
        target_column (str): The name of the target column in the data.

    Returns:
        model: Fitted ARIMA model.
    """
    logger.info(f"Training ARIMA model with order {order}")
    model = ARIMA(data[target_column], order=order)
    fitted_model = model.fit()
    logger.info("ARIMA model training complete.")
    return fitted_model


def train_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), target_column='value'):
    """
    Train a SARIMA model on the given time series data.
    
    Args:
        data (pd.DataFrame): Time series data with a target column.
        order (tuple): The (p, d, q) order of the SARIMA model.
        seasonal_order (tuple): The (P, D, Q, s) seasonal order of the SARIMA model.
        target_column (str): The name of the target column in the data.

    Returns:
        model: Fitted SARIMA model.
    """
    logger.info(f"Training SARIMA model with order {order} and seasonal_order {seasonal_order}")
    model = SARIMAX(data[target_column], order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    logger.info("SARIMA model training complete.")
    return fitted_model


def train_prophet_model(data, date_column='ds', target_column='y'):
    """
    Train a Prophet model on the given time series data.
    
    Args:
        data (pd.DataFrame): Time series data with 'ds' as date column and 'y' as target column.
        date_column (str): The name of the date column in the data.
        target_column (str): The name of the target column in the data.

    Returns:
        model: Fitted Prophet model.
    """
    logger.info("Training Prophet model")
    df = data.rename(columns={date_column: 'ds', target_column: 'y'})
    model = Prophet()
    model.fit(df)
    logger.info("Prophet model training complete.")
    return model


def evaluate_model(fitted_model, test_data, target_column='value'):
    """
    Evaluate the performance of a fitted model on test data.
    
    Args:
        fitted_model: The fitted time series model.
        test_data (pd.DataFrame): Test data with the target column.
        target_column (str): The name of the target column in the test data.

    Returns:
        dict: Evaluation metrics (e.g., MAPE, MAE).
    """
    logger.info("Evaluating model")
    predictions = fitted_model.forecast(len(test_data))
    actuals = test_data[target_column].values
    mae = (abs(predictions - actuals)).mean()
    mape = (abs((predictions - actuals) / actuals)).mean() * 100

    logger.info(f"Evaluation complete: MAE={mae}, MAPE={mape}%")
    return {"MAE": mae, "MAPE": mape}


if __name__ == "__main__":
    # Example usage
    logger.info("Loading data")
    # Replace 'path_to_data.csv' with the actual path
    data = pd.read_csv("path_to_data.csv")
    
    # Ensure data has datetime index for ARIMA/SARIMA or 'ds' and 'y' for Prophet
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # ARIMA model
    arima_model = train_arima_model(data, order=(5, 1, 0), target_column='value')
    arima_eval = evaluate_model(arima_model, data[-30:], target_column='value')
    print("ARIMA Evaluation:", arima_eval)

    # SARIMA model
    sarima_model = train_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), target_column='value')
    sarima_eval = evaluate_model(sarima_model, data[-30:], target_column='value')
    print("SARIMA Evaluation:", sarima_eval)

    # Prophet model
    prophet_data = data.reset_index()[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
    prophet_model = train_prophet_model(prophet_data)
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)
    print("Prophet Forecast:", forecast[['ds', 'yhat']].tail(10))
