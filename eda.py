"""
Exploratory Data Analysis Module
Comprehensive visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class HostelEDA:
    """Comprehensive EDA for Hostel Price Prediction"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def overview(self):
        """Display dataset overview"""
        print("\n" + "="*70)
        print("📊 HOSTEL DATASET OVERVIEW")
        print("="*70)
        
        print(f"\n📐 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        print("\n🔢 Data Types:")
        print(self.df.dtypes.value_counts())
        
        print("\n❓ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values ✅")
        
        print("\n📈 Target Variable (price_per_night) Statistics:")
        print(self.df['price_per_night'].describe())
        
        return self.df.head()
    
    def plot_price_distribution(self, save_path=None):
        """Plot price distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.df['price_per_night'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Price per Night ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(self.df['price_per_night'].mean(), color='red', linestyle='--', label=f'Mean: ${self.df["price_per_night"].mean():.2f}')
        axes[0].axvline(self.df['price_per_night'].median(), color='green', linestyle='--', label=f'Median: ${self.df["price_per_night"].median():.2f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.df['price_per_night'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1].set_ylabel('Price per Night ($)')
        axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, save_path=None):
        """Plot correlation heatmap"""
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        # Plot
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top correlations with price
        if 'price_per_night' in corr_matrix.columns:
            print("\n🔗 Top 15 Features Correlated with Price:")
            price_corr = corr_matrix['price_per_night'].sort_values(ascending=False)[1:16]
            for feature, corr in price_corr.items():
                print(f"  {feature:40s}: {corr:6.3f}")
    
    def plot_categorical_analysis(self, save_path=None):
        """Analyze categorical features"""
        categorical_cols = ['city', 'room_type', 'season']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, col in enumerate(categorical_cols):
            if col in self.df.columns:
                avg_price = self.df.groupby(col)['price_per_night'].mean().sort_values(ascending=False)
                
                axes[idx].bar(range(len(avg_price)), avg_price.values, color='coral', alpha=0.7, edgecolor='black')
                axes[idx].set_xticks(range(len(avg_price)))
                axes[idx].set_xticklabels(avg_price.index, rotation=45, ha='right')
                axes[idx].set_ylabel('Average Price ($)')
                axes[idx].set_title(f'Average Price by {col.title()}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_relationships(self, save_path=None):
        """Plot key feature relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Rating vs Price
        axes[0, 0].scatter(self.df['rating'], self.df['price_per_night'], alpha=0.5, s=30, color='blue')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Price per Night ($)')
        axes[0, 0].set_title('Rating vs Price', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distance to Center vs Price
        axes[0, 1].scatter(self.df['distance_to_center_km'], self.df['price_per_night'], alpha=0.5, s=30, color='green')
        axes[0, 1].set_xlabel('Distance to Center (km)')
        axes[0, 1].set_ylabel('Price per Night ($)')
        axes[0, 1].set_title('Distance vs Price', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Occupancy Rate vs Price
        axes[1, 0].scatter(self.df['occupancy_rate'], self.df['price_per_night'], alpha=0.5, s=30, color='red')
        axes[1, 0].set_xlabel('Occupancy Rate')
        axes[1, 0].set_ylabel('Price per Night ($)')
        axes[1, 0].set_title('Occupancy vs Price', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Beds in Room vs Price
        if 'beds_in_room' in self.df.columns:
            avg_price_beds = self.df.groupby('beds_in_room')['price_per_night'].mean()
            axes[1, 1].plot(avg_price_beds.index, avg_price_beds.values, marker='o', linewidth=2, markersize=8, color='purple')
            axes[1, 1].set_xlabel('Beds in Room')
            axes[1, 1].set_ylabel('Average Price ($)')
            axes[1, 1].set_title('Beds vs Average Price', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_amenities_impact(self, save_path=None):
        """Analyze impact of amenities on price"""
        amenities = ['breakfast_included', 'wifi', 'laundry', 'kitchen_access', 
                    'air_conditioning', 'locker_available', 'security_24h']
        
        available_amenities = [a for a in amenities if a in self.df.columns]
        
        if len(available_amenities) == 0:
            print("No amenity features found")
            return
        
        amenity_impact = {}
        for amenity in available_amenities:
            with_amenity = self.df[self.df[amenity] == 1]['price_per_night'].mean()
            without_amenity = self.df[self.df[amenity] == 0]['price_per_night'].mean()
            amenity_impact[amenity] = with_amenity - without_amenity
        
        # Sort by impact
        sorted_impact = dict(sorted(amenity_impact.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(12, 6))
        colors = ['green' if v > 0 else 'red' for v in sorted_impact.values()]
        plt.bar(range(len(sorted_impact)), sorted_impact.values(), color=colors, alpha=0.7, edgecolor='black')
        plt.xticks(range(len(sorted_impact)), [k.replace('_', ' ').title() for k in sorted_impact.keys()], rotation=45, ha='right')
        plt.ylabel('Price Impact ($)')
        plt.title('Impact of Amenities on Price', fontsize=14, fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n💰 Amenity Price Impact:")
        for amenity, impact in sorted_impact.items():
            print(f"  {amenity:30s}: ${impact:+6.2f}")
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        # 1. 3D Scatter: Rating, Distance, Price
        fig1 = px.scatter_3d(self.df, x='rating', y='distance_to_center_km', z='price_per_night',
                            color='price_per_night', size='num_reviews',
                            title='3D View: Rating, Distance, and Price',
                            labels={'price_per_night': 'Price ($)', 
                                   'distance_to_center_km': 'Distance (km)'},
                            color_continuous_scale='Viridis')
        fig1.show()
        
        # 2. Price by City
        if 'city' in self.df.columns:
            city_stats = self.df.groupby('city').agg({
                'price_per_night': ['mean', 'median', 'count']
            }).reset_index()
            city_stats.columns = ['city', 'mean_price', 'median_price', 'count']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=city_stats['city'], y=city_stats['mean_price'], 
                                 name='Mean Price', marker_color='lightblue'))
            fig2.add_trace(go.Bar(x=city_stats['city'], y=city_stats['median_price'],
                                 name='Median Price', marker_color='coral'))
            fig2.update_layout(title='Price Comparison by City', 
                             xaxis_title='City', yaxis_title='Price ($)',
                             barmode='group')
            fig2.show()
    
    def generate_full_report(self, output_dir='hostel_price_prediction/eda_plots'):
        """Generate complete EDA report with all visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n🎨 Generating EDA Report...")
        
        self.overview()
        self.plot_price_distribution(f'{output_dir}/price_distribution.png')
        self.plot_correlation_heatmap(f'{output_dir}/correlation_heatmap.png')
        self.plot_categorical_analysis(f'{output_dir}/categorical_analysis.png')
        self.plot_feature_relationships(f'{output_dir}/feature_relationships.png')
        self.plot_amenities_impact(f'{output_dir}/amenities_impact.png')
        
        print(f"\n✅ EDA Report saved to: {output_dir}/")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('hostel_price_prediction/data/hostel_data.csv')
    
    # Run EDA
    eda = HostelEDA(df)
    eda.generate_full_report()
