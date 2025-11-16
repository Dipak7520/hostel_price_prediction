"""
Data Preprocessing Module for Hostel Price Prediction
Handles data cleaning, encoding, scaling, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class HostelDataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def clean_data(self, df):
        """Clean and handle missing values"""
        print("🧹 Cleaning data...")
        
        # Create a copy
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"  Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle outliers in price (3 standard deviations)
        if 'price_per_night' in df.columns:
            price_mean = df['price_per_night'].mean()
            price_std = df['price_per_night'].std()
            df = df[np.abs(df['price_per_night'] - price_mean) <= 3 * price_std]
        
        print(f"  Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def engineer_features(self, df):
        """Create advanced engineered features"""
        print("⚙️ Engineering features...")
        
        df = df.copy()
        
        # 1. Demand Index
        df['demand_index'] = (
            df['occupancy_rate'] * 0.4 + 
            (df['rating'] / 10) * 0.3 + 
            (df['num_reviews'] / df['num_reviews'].max()) * 0.2 +
            df['season'].map({'low': 0.3, 'shoulder': 0.7, 'peak': 1.0}) * 0.1
        )
        
        # 2. Interaction Features
        df['rating_reviews_interaction'] = df['rating'] * np.log1p(df['num_reviews'])
        df['distance_season_interaction'] = df['distance_to_center_km'] * df['season'].map({'low': 1, 'shoulder': 2, 'peak': 3})
        df['price_occupancy_ratio'] = df['occupancy_rate'] / (df['distance_to_center_km'] + 1)
        
        # 3. Room Density
        df['room_density'] = df['beds_in_room'] / df['room_area_sqft']
        
        # 4. Amenities Score
        df['amenities_score'] = (
            df['breakfast_included'] +
            df['wifi'] +
            df['laundry'] +
            df['kitchen_access'] +
            df['air_conditioning'] +
            df['locker_available'] +
            df['security_24h']
        )
        
        # 5. Location Quality Score
        df['location_quality'] = (
            (10 - df['distance_to_center_km']) * 0.3 +
            df['walkability_score'] / 10 * 0.3 +
            df['neighbourhood_popularity_score'] * 0.2 +
            (10 - df['noise_level_index']) * 0.1 +
            (5 - df['distance_to_transport_hub_km']) * 0.1
        )
        
        # 6. Popularity Index
        df['popularity_index'] = np.log1p(df['num_reviews']) * df['rating'] / 10
        
        # 7. Value Score
        df['value_score'] = df['amenities_score'] / (df['hostel_age_years'] + 1)
        
        # 8. Weekend Premium
        df['weekend_premium'] = df['weekend'] * df['occupancy_rate']
        
        # 9. Sentiment Quality
        df['sentiment_quality'] = df['sentiment_score'] * df['rating']
        
        # 10. Booking Urgency
        df['booking_urgency'] = np.where(df['days_until_checkin'] < 7, 1, 0)
        df['last_minute_multiplier'] = np.exp(-df['days_until_checkin'] / 30)
        
        print(f"  Created {df.shape[1] - len(df.columns) + 12} new features")
        return df
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        print("🔤 Encoding categorical variables...")
        
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if df[col].nunique() <= 10:  # One-hot encode if few categories
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)
                else:  # Label encode if many categories
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    df.drop(col, axis=1, inplace=True)
        
        return df
    
    def add_clustering_features(self, df, n_clusters=5):
        """Add KMeans clustering labels as features"""
        print(f"🎯 Creating {n_clusters} hostel clusters...")
        
        df = df.copy()
        
        # Select features for clustering
        cluster_features = ['rating', 'num_reviews', 'distance_to_center_km', 
                           'occupancy_rate', 'neighbourhood_popularity_score',
                           'amenities_score']
        
        # Ensure features exist
        cluster_features = [f for f in cluster_features if f in df.columns]
        
        if len(cluster_features) > 0:
            X_cluster = df[cluster_features].fillna(0)
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster_label'] = kmeans.fit_predict(X_scaled)
            
            # Create cluster dummy variables
            cluster_dummies = pd.get_dummies(df['cluster_label'], prefix='cluster')
            df = pd.concat([df, cluster_dummies], axis=1)
            
            print(f"  Created cluster labels: {df['cluster_label'].value_counts().to_dict()}")
        
        return df
    
    def prepare_for_modeling(self, df, target_col='price_per_night'):
        """Prepare final dataset for modeling"""
        print("🎯 Preparing data for modeling...")
        
        df = df.copy()
        
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(target_col, axis=1)
        else:
            y = None
            X = df
        
        # Remove non-numeric columns (like hostel_id if exists)
        non_numeric = ['hostel_id', 'latitude', 'longitude']  # Keep for reference but don't use in model
        X = X.select_dtypes(include=[np.number])
        
        # Handle any remaining NaN
        X = X.fillna(0)
        
        print(f"  Final feature count: {X.shape[1]}")
        print(f"  Sample size: {X.shape[0]}")
        
        return X, y
    
    def get_feature_summary(self, df):
        """Get comprehensive feature summary"""
        summary = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        return summary


def preprocess_pipeline(df):
    """Complete preprocessing pipeline"""
    print("\n" + "="*60)
    print("🚀 HOSTEL DATA PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    preprocessor = HostelDataPreprocessor()
    
    # Step 1: Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Step 2: Engineer features
    df_engineered = preprocessor.engineer_features(df_clean)
    
    # Step 3: Encode categorical variables
    categorical_cols = ['city', 'room_type', 'season', 'topic_category', 'bathroom_type']
    df_encoded = preprocessor.encode_categorical(df_engineered, categorical_cols)
    
    # Step 4: Add clustering features
    df_clustered = preprocessor.add_clustering_features(df_encoded, n_clusters=5)
    
    # Step 5: Prepare for modeling
    X, y = preprocessor.prepare_for_modeling(df_clustered)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Target Vector Shape: {y.shape}")
    print(f"\nFeature Names: {list(X.columns[:10])}... (showing first 10)")
    
    return X, y, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    df = pd.read_csv('hostel_price_prediction/data/hostel_data.csv')
    X, y, preprocessor = preprocess_pipeline(df)
    
    # Save processed data
    X.to_csv('hostel_price_prediction/data/X_processed.csv', index=False)
    y.to_csv('hostel_price_prediction/data/y_processed.csv', index=False)
    print("\n💾 Processed data saved!")
