"""
Hostel Price Prediction - Synthetic Data Generator
Generates comprehensive hostel dataset with basic and advanced features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

class HostelDataGenerator:
    """Generate realistic synthetic hostel data with advanced features"""
    
    def __init__(self, n_samples=2000):
        self.n_samples = n_samples
        self.cities = ['Mumbai', 'Delhi', 'Bangalore', 'Goa', 'Jaipur',
                      'Kolkata', 'Chennai', 'Hyderabad', 'Pune', 'Udaipur',
                      'Varanasi', 'Rishikesh', 'Manali', 'Agra', 'Kerala']
        
        self.room_types = ['shared', 'private']
        self.seasons = ['low', 'shoulder', 'peak']
        
    def generate_dataset(self):
        """Generate complete hostel dataset"""
        print("🏨 Generating Hostel Price Prediction Dataset...")
        
        data = {
            # Basic Features
            'hostel_id': range(1, self.n_samples + 1),
            'city': np.random.choice(self.cities, self.n_samples),
            'distance_to_center_km': np.round(np.random.exponential(3, self.n_samples), 2),
            'rating': np.round(np.random.uniform(6.0, 9.8, self.n_samples), 1),
            'num_reviews': np.random.randint(10, 2000, self.n_samples),
            'room_type': np.random.choice(self.room_types, self.n_samples, p=[0.7, 0.3]),
            'beds_in_room': np.random.choice([2, 4, 6, 8, 10, 12], self.n_samples, p=[0.05, 0.25, 0.3, 0.25, 0.1, 0.05]),
            'breakfast_included': np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4]),
            'wifi': np.random.choice([0, 1], self.n_samples, p=[0.05, 0.95]),
            'laundry': np.random.choice([0, 1], self.n_samples, p=[0.4, 0.6]),
            'season': np.random.choice(self.seasons, self.n_samples),
            'weekend': np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3]),
            'occupancy_rate': np.round(np.random.uniform(0.4, 1.0, self.n_samples), 2),
        }
        
        df = pd.DataFrame(data)
        
        # Advanced Features
        df = self._add_geospatial_features(df)
        df = self._add_nlp_features(df)
        df = self._add_demand_features(df)
        df = self._add_time_features(df)
        df = self._add_property_features(df)
        
        # Calculate target price
        df['price_per_night'] = self._calculate_price(df)
        
        print(f"✅ Generated {len(df)} hostel records with {len(df.columns)} features")
        return df
    
    def _add_geospatial_features(self, df):
        """Add geospatial and location-based features"""
        # Latitude/Longitude for Indian cities
        city_coords = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Goa': (15.2993, 74.1240),
            'Jaipur': (26.9124, 75.7873),
            'Kolkata': (22.5726, 88.3639),
            'Chennai': (13.0827, 80.2707),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Udaipur': (24.5854, 73.7125),
            'Varanasi': (25.3176, 82.9739),
            'Rishikesh': (30.0869, 78.2676),
            'Manali': (32.2396, 77.1887),
            'Agra': (27.1767, 78.0081),
            'Kerala': (10.8505, 76.2711)
        }
        
        df['latitude'] = df['city'].map(lambda x: city_coords[x][0] + np.random.uniform(-0.1, 0.1))
        df['longitude'] = df['city'].map(lambda x: city_coords[x][1] + np.random.uniform(-0.1, 0.1))
        df['distance_to_transport_hub_km'] = np.round(np.random.exponential(2, len(df)), 2)
        df['distance_to_popular_places'] = np.round(np.random.exponential(1.5, len(df)), 2)
        df['neighbourhood_popularity_score'] = np.round(np.random.uniform(3, 10, len(df)), 1)
        df['walkability_score'] = np.round(np.random.uniform(40, 100, len(df)), 0)
        df['noise_level_index'] = np.round(np.random.uniform(1, 10, len(df)), 1)
        
        return df
    
    def _add_nlp_features(self, df):
        """Add NLP-based features (simulated from reviews)"""
        df['sentiment_score'] = np.round(np.random.uniform(-0.2, 1.0, len(df)), 2)
        df['review_keywords_count'] = np.random.randint(5, 50, len(df))
        df['topic_category'] = np.random.choice(['cleanliness', 'staff', 'location', 'noise', 'value'], len(df))
        df['recent_sentiment_trend'] = np.round(np.random.uniform(-0.3, 0.3, len(df)), 2)
        
        return df
    
    def _add_demand_features(self, df):
        """Add dynamic pricing and demand indicators"""
        df['days_until_checkin'] = np.random.randint(0, 90, len(df))
        df['surge_pricing_flag'] = np.random.choice([0, 1], len(df), p=[0.75, 0.25])
        df['competitor_average_price'] = np.round(np.random.uniform(300, 1500, len(df)), 2)
        df['competitor_availability'] = np.round(np.random.uniform(0.1, 0.9, len(df)), 2)
        df['price_last_week'] = np.round(np.random.uniform(300, 1400, len(df)), 2)
        
        return df
    
    def _add_time_features(self, df):
        """Add time-based features"""
        df['month'] = np.random.randint(1, 13, len(df))
        df['holiday_or_festival'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
        df['special_event_flag'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
        
        # Seasonal index based on month
        seasonal_map = {1: 0.7, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 1.1,
                       7: 1.2, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.9}
        df['seasonal_index'] = df['month'].map(seasonal_map)
        
        return df
    
    def _add_property_features(self, df):
        """Add hostel property features"""
        df['hostel_age_years'] = np.random.randint(1, 50, len(df))
        df['kitchen_access'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        df['air_conditioning'] = np.random.choice([0, 1], len(df), p=[0.5, 0.5])
        df['locker_available'] = np.random.choice([0, 1], len(df), p=[0.2, 0.8])
        df['security_24h'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        df['female_only_dorm_available'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
        df['common_area_count'] = np.random.randint(1, 6, len(df))
        df['room_area_sqft'] = np.random.randint(80, 400, len(df))
        df['bathroom_type'] = np.random.choice(['shared', 'ensuite'], len(df), p=[0.7, 0.3])
        
        return df
    
    def _calculate_price(self, df):
        """Calculate price in INR based on features with realistic Indian hostel pricing"""
        base_price = 300  # Base price in INR
        
        # City premium in INR (realistic Indian hostel prices)
        city_premium = {
            'Mumbai': 500, 'Delhi': 450, 'Bangalore': 400, 'Goa': 600, 'Manali': 550,
            'Jaipur': 350, 'Hyderabad': 350, 'Pune': 380, 'Chennai': 340, 'Kolkata': 320,
            'Udaipur': 400, 'Varanasi': 280, 'Rishikesh': 350, 'Agra': 300, 'Kerala': 450
        }
        price = base_price + df['city'].map(city_premium)
        
        # Room type impact (private rooms cost more in India)
        price += np.where(df['room_type'] == 'private', 400, 0)
        
        # Beds in room (more beds = cheaper per person) - non-linear relationship
        price -= (df['beds_in_room'] - 2) * 30 - (df['beds_in_room'] ** 1.3) * 5
        
        # Distance penalty (INR per km) - exponential decay
        price -= df['distance_to_center_km'] * 20 + (df['distance_to_center_km'] ** 1.5) * 5
        
        # Rating premium - polynomial relationship
        price += (df['rating'] - 7) * 80 + ((df['rating'] - 7) ** 2) * 10
        
        # Reviews trust factor - logarithmic with threshold effects
        review_factor = np.log1p(df['num_reviews']) * 15
        review_factor += np.where(df['num_reviews'] > 500, 50, 0)  # Bonus for very popular
        price += review_factor
        
        # Amenities (Indian hostel context) - interaction effects
        amenity_score = (
            df['breakfast_included'] * 80 + 
            df['air_conditioning'] * 100 +  # AC is premium in India
            df['locker_available'] * 30 + 
            df['kitchen_access'] * 50
        )
        # Bonus if multiple premium amenities
        premium_count = df['breakfast_included'] + df['air_conditioning']
        amenity_score += np.where(premium_count >= 2, 80, 0)
        price += amenity_score
        
        # Season and demand (India-specific pricing) - multiplicative effects
        season_multiplier = df['season'].map({'low': 0.75, 'shoulder': 1.0, 'peak': 1.4})
        price *= season_multiplier
        
        # Weekend and occupancy interaction
        weekend_boost = df['weekend'] * 100 * (1 + df['occupancy_rate'] * 0.5)
        price += weekend_boost
        
        price += df['occupancy_rate'] * 200
        price += df['surge_pricing_flag'] * 150 * (1 + df['occupancy_rate'])
        
        # Location quality - non-linear
        location_score = df['neighbourhood_popularity_score'] * 20
        location_score += (df['walkability_score'] / 100) ** 2 * 150
        location_score -= df['noise_level_index'] * 15
        price += location_score
        
        # Sentiment impact - exponential for very high sentiment
        sentiment_boost = df['sentiment_score'] * 100
        sentiment_boost += np.where(df['sentiment_score'] > 0.8, 80, 0)
        price += sentiment_boost
        
        # Special events (festivals, holidays - big impact in India)
        event_multiplier = 1.0 + (df['holiday_or_festival'] * 0.15) + (df['special_event_flag'] * 0.2)
        price *= event_multiplier
        
        # Seasonal index
        price *= df['seasonal_index']
        
        # Add some random noise with moderate heteroscedasticity
        noise_scale = 50 + (price / 1500) * 30  # Increased noise for target R² range
        price += np.random.normal(0, noise_scale, len(df))
        
        # Ensure prices are realistic for Indian hostels (200-3000 INR)
        price = np.maximum(price, 200)
        price = np.minimum(price, 3000)
        return np.round(price, 2)


def main():
    """Generate and save hostel dataset"""
    generator = HostelDataGenerator(n_samples=2000)
    df = generator.generate_dataset()
    
    # Save dataset
    output_path = 'data/hostel_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Display summary
    print("\n📊 Dataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nPrice Statistics:")
    print(df['price_per_night'].describe())
    print(f"\nFeature Types:")
    print(df.dtypes.value_counts())
    
    return df


if __name__ == "__main__":
    main()
