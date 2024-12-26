import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class InventoryPredictor:
    def __init__(self):
        self.sales_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.demand_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    def prepare_features(self, historical_data):
        """Enhanced feature engineering"""
        df = pd.DataFrame(historical_data)
        
        # Time-based features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Sales and price features
        df['price_normalized'] = df['price'] / df['price'].mean()
        df['discount_factor'] = 1 - (df['discount'] / 100)
        df['effective_price'] = df['price'] * df['discount_factor']
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}d'] = df['units_sold'].rolling(window).mean()
            df[f'rolling_std_{window}d'] = df['units_sold'].rolling(window).std()
            df[f'rolling_max_{window}d'] = df['units_sold'].rolling(window).max()
            
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}d'] = df['units_sold'].shift(lag)
            
        # Stock level indicators
        df['stock_ratio'] = df['stock'] / df['units_sold'].rolling(7).mean()
        df['days_to_stockout'] = df['stock'] / df['units_sold'].rolling(7).mean()
        
        # Lead time impact
        df['lead_time_coverage'] = df['stock'] / (df['units_sold'].rolling(7).mean() * df['lead_time'])
        
        return df.fillna(method='bfill')

    def train_models(self, historical_data):
        """Train both sales and demand models"""
        df = self.prepare_features(historical_data)
        
        feature_columns = [
            'day_of_week', 'month', 'quarter', 'is_weekend', 'is_month_end',
            'price_normalized', 'discount_factor', 'effective_price', 'lead_time',
            'rolling_mean_7d', 'rolling_std_7d', 'rolling_max_7d',
            'rolling_mean_30d', 'rolling_std_30d', 'rolling_max_30d',
            'lag_1d', 'lag_7d', 'stock_ratio', 'lead_time_coverage'
        ]
        
        X = df[feature_columns].values
        y_sales = df['units_sold'].values
        y_demand = df['stock_ratio'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.sales_model.fit(X_scaled, y_sales)
        self.demand_model.fit(X_scaled, y_demand)
        
        # Train clustering model for demand patterns
        self.kmeans.fit(df[['units_sold', 'stock_ratio', 'lead_time_coverage']])
        
        return {
            'sales_score': self.sales_model.score(X_scaled, y_sales),
            'demand_score': self.demand_model.score(X_scaled, y_demand)
        }

    def predict_inventory(self, product_data, forecast_days=7):
        """Generate comprehensive inventory predictions"""
        df = self.prepare_features(product_data)
        current_stock = df['stock'].iloc[-1]
        current_price = df['price'].iloc[-1]
        lead_time = df['lead_time'].iloc[-1]
        
        # Prepare future dates
        last_date = df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
        
        # Create future features
        future_df = pd.DataFrame({'date': future_dates})
        future_df = self.prepare_features(future_df)
        
        # Make predictions
        X_future = self.scaler.transform(future_df[self.feature_columns].values)
        sales_pred = self.sales_model.predict(X_future)
        demand_pred = self.demand_model.predict(X_future)
        
        # Calculate confidence intervals using model's feature importances
        sales_std = np.std([tree.predict(X_future) 
                           for tree in self.sales_model.estimators_], axis=0)
        
        # Calculate inventory metrics
        cumulative_sales = np.cumsum(sales_pred)
        projected_stock = current_stock - cumulative_sales
        reorder_point = self._calculate_reorder_point(
            sales_pred, lead_time, safety_factor=1.5
        )
        
        # Determine optimal reorder quantity
        optimal_order = self._calculate_optimal_order(
            sales_pred, current_stock, lead_time, 
            holding_cost=0.1, stockout_cost=0.5
        )
        
        # Cluster current demand pattern
        demand_pattern = self.kmeans.predict([[
            np.mean(sales_pred), 
            current_stock / np.mean(sales_pred),
            current_stock / (np.mean(sales_pred) * lead_time)
        ]])[0]
        
        return {
            'dates': future_dates,
            'sales_forecast': sales_pred.tolist(),
            'confidence_intervals': {
                'lower': (sales_pred - 1.96 * sales_std).tolist(),
                'upper': (sales_pred + 1.96 * sales_std).tolist()
            },
            'stock_projection': projected_stock.tolist(),
            'reorder_point': reorder_point,
            'optimal_order_quantity': optimal_order,
            'stockout_probability': self._calculate_stockout_probability(
                projected_stock, sales_std
            ),
            'demand_pattern': {
                'cluster': demand_pattern,
                'pattern_type': ['Stable', 'Volatile', 'Seasonal'][demand_pattern],
                'confidence': self._calculate_pattern_confidence(demand_pattern)
            },
            'recommendations': self._generate_recommendations(
                current_stock, reorder_point, optimal_order, 
                lead_time, sales_pred, demand_pattern
            )
        }

    def _calculate_reorder_point(self, sales_forecast, lead_time, safety_factor=1.5):
        """Calculate reorder point with safety stock"""
        daily_demand = np.mean(sales_forecast)
        demand_std = np.std(sales_forecast)
        safety_stock = safety_factor * demand_std * np.sqrt(lead_time)
        return (daily_demand * lead_time) + safety_stock

    def _calculate_optimal_order(self, sales_forecast, current_stock, lead_time, 
                               holding_cost=0.1, stockout_cost=0.5):
        """Calculate optimal order quantity using newsvendor model"""
        daily_demand = np.mean(sales_forecast)
        demand_std = np.std(sales_forecast)
        service_level = stockout_cost / (holding_cost + stockout_cost)
        z_score = stats.norm.ppf(service_level)
        
        optimal_quantity = (daily_demand * lead_time) + (z_score * demand_std * np.sqrt(lead_time))
        return max(0, optimal_quantity - current_stock)

    def _calculate_stockout_probability(self, projected_stock, sales_std):
        """Calculate probability of stockout"""
        return stats.norm.cdf(0, loc=projected_stock, scale=sales_std)

    def _calculate_pattern_confidence(self, cluster_id):
        """Calculate confidence in demand pattern classification"""
        distances = self.kmeans.transform([[
            np.mean(sales_pred), 
            current_stock / np.mean(sales_pred),
            current_stock / (np.mean(sales_pred) * lead_time)
        ]])
        return 1 / (1 + distances[0][cluster_id])

    def _generate_recommendations(self, current_stock, reorder_point, optimal_order, 
                                lead_time, sales_forecast, demand_pattern):
        """Generate actionable recommendations"""
        recommendations = []
        
        if current_stock <= reorder_point:
            recommendations.append({
                'type': 'urgent',
                'message': f'Place order for {optimal_order:.0f} units immediately',
                'reason': 'Stock below reorder point'
            })
        
        avg_daily_sales = np.mean(sales_forecast)
        days_of_stock = current_stock / avg_daily_sales
        
        if days_of_stock < lead_time * 1.5:
            recommendations.append({
                'type': 'warning',
                'message': 'Stock levels critical considering lead time',
                'reason': f'Only {days_of_stock:.1f} days of stock remaining'
            })
        
        pattern_recommendations = {
            0: 'Consider automated reordering for stable demand',
            1: 'Implement safety stock due to volatile demand',
            2: 'Plan inventory around seasonal patterns'
        }
        
        recommendations.append({
            'type': 'strategy',
            'message': pattern_recommendations[demand_pattern],
            'reason': f'Based on {["stable", "volatile", "seasonal"][demand_pattern]} demand pattern'
        })
        
        return recommendations 