"""
Flask Backend API for Hostel Price Prediction
Production-ready REST API with comprehensive error handling
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'hostel_price_prediction', 'models', 'best_model.pkl')
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'data', 'X_processed.csv')

# Global variables
model = None
feature_names = None
feature_stats = None

def load_model():
    """Load the trained ML model"""
    global model, feature_names, feature_stats
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
        
        # Load feature names and statistics
        X_sample = pd.read_csv(FEATURE_NAMES_PATH)
        feature_names = X_sample.columns.tolist()
        feature_stats = {
            'mean': X_sample.mean().to_dict(),
            'std': X_sample.std().to_dict(),
            'min': X_sample.min().to_dict(),
            'max': X_sample.max().to_dict()
        }
        logger.info(f"✅ Loaded {len(feature_names)} feature names and statistics")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

def create_feature_vector(user_input):
    """
    Create a complete feature vector from user input
    Handles missing features with intelligent defaults
    """
    try:
        # Initialize with default values (median/mode)
        features = {}
        
        # Map user input to features
        city_mapping = {
            'Mumbai': 0, 'Delhi': 1, 'Bangalore': 2, 'Goa': 3, 'Jaipur': 4,
            'Kolkata': 5, 'Chennai': 6, 'Hyderabad': 7, 'Pune': 8, 'Udaipur': 9,
            'Varanasi': 10, 'Rishikesh': 11, 'Manali': 12, 'Agra': 13, 'Kerala': 14
        }
        
        room_type_mapping = {'shared': 0, 'private': 1}
        season_mapping = {'low': 0, 'shoulder': 1, 'peak': 2}
        bathroom_mapping = {'shared': 0, 'ensuite': 1}
        
        # Core features from user input
        features['hostel_id'] = 1
        features['distance_to_center_km'] = float(user_input.get('distance_to_center', 2.0))
        features['rating'] = float(user_input.get('rating', 8.0))
        features['num_reviews'] = int(user_input.get('num_reviews', 100))
        features['beds_in_room'] = int(user_input.get('beds_in_room', 6))
        features['breakfast_included'] = int(user_input.get('breakfast_included', 0))
        features['wifi'] = int(user_input.get('wifi', 1))
        features['laundry'] = int(user_input.get('laundry', 1))
        features['weekend'] = int(user_input.get('weekend', 0))
        features['occupancy_rate'] = float(user_input.get('occupancy_rate', 0.7))
        
        # Geospatial features
        city = user_input.get('city', 'Mumbai')
        city_coords = {
            'Mumbai': (19.0760, 72.8777), 'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946), 'Goa': (15.2993, 74.1240),
            'Jaipur': (26.9124, 75.7873), 'Kolkata': (22.5726, 88.3639),
            'Chennai': (13.0827, 80.2707), 'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567), 'Udaipur': (24.5854, 73.7125),
            'Varanasi': (25.3176, 82.9739), 'Rishikesh': (30.0869, 78.2676),
            'Manali': (32.2396, 77.1887), 'Agra': (27.1767, 78.0081),
            'Kerala': (10.8505, 76.2711)
        }
        
        coords = city_coords.get(city, (19.0760, 72.8777))
        features['latitude'] = coords[0]
        features['longitude'] = coords[1]
        features['distance_to_transport_hub_km'] = float(user_input.get('distance_to_transport', 1.5))
        features['distance_to_popular_places'] = float(user_input.get('distance_to_attractions', 1.0))
        features['neighbourhood_popularity_score'] = float(user_input.get('neighbourhood_score', 7.0))
        features['walkability_score'] = float(user_input.get('walkability', 75))
        features['noise_level_index'] = float(user_input.get('noise_level', 5.0))
        
        # NLP features
        features['sentiment_score'] = float(user_input.get('sentiment_score', 0.7))
        features['review_keywords_count'] = int(user_input.get('keywords_count', 25))
        features['recent_sentiment_trend'] = float(user_input.get('sentiment_trend', 0.0))
        
        # Demand features
        features['days_until_checkin'] = int(user_input.get('days_until_checkin', 14))
        features['surge_pricing_flag'] = int(user_input.get('surge_pricing', 0))
        features['competitor_average_price'] = float(user_input.get('competitor_price', 800.0))
        features['competitor_availability'] = float(user_input.get('competitor_availability', 0.5))
        features['price_last_week'] = float(user_input.get('price_last_week', 750.0))
        
        # Time features
        features['month'] = int(user_input.get('month', 6))
        features['holiday_or_festival'] = int(user_input.get('holiday', 0))
        features['special_event_flag'] = int(user_input.get('special_event', 0))
        
        seasonal_map = {1: 0.7, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 1.1,
                       7: 1.2, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.9}
        features['seasonal_index'] = seasonal_map.get(features['month'], 1.0)
        
        # Property features
        features['hostel_age_years'] = int(user_input.get('hostel_age', 10))
        features['kitchen_access'] = int(user_input.get('kitchen', 1))
        features['air_conditioning'] = int(user_input.get('ac', 1))
        features['locker_available'] = int(user_input.get('locker', 1))
        features['security_24h'] = int(user_input.get('security', 1))
        features['female_only_dorm_available'] = int(user_input.get('female_dorm', 0))
        features['common_area_count'] = int(user_input.get('common_areas', 3))
        features['room_area_sqft'] = int(user_input.get('room_area', 200))
        
        # One-hot encoded features
        room_type = user_input.get('room_type', 'shared')
        features['room_type_private'] = 1 if room_type == 'private' else 0
        features['room_type_shared'] = 1 if room_type == 'shared' else 0
        
        season = user_input.get('season', 'shoulder')
        features['season_low'] = 1 if season == 'low' else 0
        features['season_peak'] = 1 if season == 'peak' else 0
        features['season_shoulder'] = 1 if season == 'shoulder' else 0
        
        bathroom = user_input.get('bathroom_type', 'shared')
        features['bathroom_type_ensuite'] = 1 if bathroom == 'ensuite' else 0
        features['bathroom_type_shared'] = 1 if bathroom == 'shared' else 0
        
        # Topic category one-hot
        topic = user_input.get('topic_category', 'location')
        for cat in ['cleanliness', 'location', 'noise', 'staff', 'value']:
            features[f'topic_category_{cat}'] = 1 if topic == cat else 0
        
        # City one-hot encoding
        for city_name in city_mapping.keys():
            city_safe = city_name.replace(' ', '_')
            features[f'city_{city_safe}'] = 1 if city == city_name else 0
        
        # Engineered features
        features['demand_index'] = features['occupancy_rate'] * (1 + features['surge_pricing_flag'])
        features['rating_reviews_interaction'] = features['rating'] * np.log1p(features['num_reviews'])
        features['distance_rating_interaction'] = features['distance_to_center_km'] * features['rating']
        features['room_density'] = features['room_area_sqft'] / features['beds_in_room']
        features['amenities_score'] = (features['breakfast_included'] + features['wifi'] + 
                                       features['kitchen_access'] + features['air_conditioning'] + 
                                       features['locker_available'] + features['security_24h'])
        features['location_quality'] = (features['neighbourhood_popularity_score'] * 0.6 + 
                                        features['walkability_score'] * 0.4)
        
        # Cluster placeholder (would need actual clustering model)
        features['hostel_cluster'] = 0
        
        # Create DataFrame with correct feature order
        feature_df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for fname in feature_names:
            if fname not in feature_df.columns:
                feature_df[fname] = feature_stats['mean'].get(fname, 0)
        
        # Reorder to match training data
        feature_df = feature_df[feature_names]
        
        return feature_df
        
    except Exception as e:
        logger.error(f"Error creating feature vector: {e}")
        raise

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict hostel price based on input features
    
    Expected JSON input:
    {
        "city": "Paris",
        "distance_to_center": 2.5,
        "rating": 8.5,
        "room_type": "private",
        "beds_in_room": 2,
        ...
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the server.'
            }), 500
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            }), 400
        
        logger.info(f"Received prediction request: {data.get('city', 'N/A')}, {data.get('room_type', 'N/A')}")
        
        # Create feature vector
        features = create_feature_vector(data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Ensure prediction is positive and reasonable for Indian hostels (₹200-3000)
        prediction = max(200, min(3000, abs(prediction)))
        
        # Calculate confidence interval (using model-appropriate uncertainty)
        # For Gradient Boosting: ~18% average error
        prediction_std = prediction * 0.18  # Based on MAPE of trained model
        confidence_lower = max(200, prediction - 1.96 * prediction_std)
        confidence_upper = min(3000, prediction + 1.96 * prediction_std)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'price': round(float(prediction), 2),
                'currency': 'INR',
                'confidence_interval': {
                    'lower': round(float(confidence_lower), 2),
                    'upper': round(float(confidence_upper), 2)
                }
            },
            'input_summary': {
                'city': data.get('city', 'N/A'),
                'room_type': data.get('room_type', 'N/A'),
                'rating': data.get('rating', 'N/A'),
                'beds': data.get('beds_in_room', 'N/A')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction successful: ${prediction:.2f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get available features and their options"""
    features_info = {
        'cities': ['Mumbai', 'Delhi', 'Bangalore', 'Goa', 'Jaipur',
                   'Kolkata', 'Chennai', 'Hyderabad', 'Pune', 'Udaipur',
                   'Varanasi', 'Rishikesh', 'Manali', 'Agra', 'Kerala'],
        'room_types': ['shared', 'private'],
        'seasons': ['low', 'shoulder', 'peak'],
        'bathroom_types': ['shared', 'ensuite'],
        'ranges': {
            'distance_to_center': {'min': 0, 'max': 15, 'default': 2.0},
            'rating': {'min': 6.0, 'max': 10.0, 'default': 8.0},
            'beds_in_room': {'min': 2, 'max': 12, 'default': 6},
            'occupancy_rate': {'min': 0.3, 'max': 1.0, 'default': 0.7}
        }
    }
    return jsonify(features_info)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*70)
    print("🏨 HOSTEL PRICE PREDICTION API")
    print("="*70)
    
    # Load model at startup
    if load_model():
        print(f"\n✅ Model loaded successfully")
        print(f"✅ Features: {len(feature_names)}")
        print(f"\n🚀 Starting Flask server...")
        print(f"📍 URL: http://localhost:5000")
        print(f"📍 API: http://localhost:5000/api/predict")
        print("\n" + "="*70)
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n❌ Failed to load model. Please check the model path.")
        print(f"Expected path: {MODEL_PATH}")
