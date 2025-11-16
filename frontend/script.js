// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const form = document.getElementById('predictionForm');
const resultsPanel = document.getElementById('resultsPanel');
const placeholder = document.getElementById('placeholder');
const resultsContent = document.getElementById('results');
const loadingState = document.getElementById('loading');
const errorState = document.getElementById('error');
const occupancyRange = document.getElementById('occupancy_rate');
const occupancyDisplay = document.getElementById('occupancy_display');

// Update occupancy display
occupancyRange.addEventListener('input', (e) => {
    const value = (parseFloat(e.target.value) * 100).toFixed(0);
    occupancyDisplay.textContent = `${value}%`;
});

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading state
    showLoading();
    
    // Collect form data
    const formData = new FormData(form);
    const data = {
        city: formData.get('city'),
        room_type: formData.get('room_type'),
        distance_to_center: parseFloat(formData.get('distance_to_center')),
        rating: parseFloat(formData.get('rating')),
        beds_in_room: parseInt(formData.get('beds_in_room')),
        num_reviews: parseInt(formData.get('num_reviews')),
        breakfast_included: formData.get('breakfast_included') ? 1 : 0,
        wifi: formData.get('wifi') ? 1 : 0,
        kitchen: formData.get('kitchen') ? 1 : 0,
        ac: formData.get('ac') ? 1 : 0,
        locker: formData.get('locker') ? 1 : 0,
        security: formData.get('security') ? 1 : 0,
        season: formData.get('season'),
        month: parseInt(formData.get('month')),
        occupancy_rate: parseFloat(formData.get('occupancy_rate')),
        weekend: formData.get('weekend') ? 1 : 0,
        holiday: formData.get('holiday') ? 1 : 0,
        laundry: 1,
        bathroom_type: 'shared',
        distance_to_transport: 1.5,
        distance_to_attractions: 1.0,
        neighbourhood_score: 7.0,
        walkability: 75,
        noise_level: 5.0,
        sentiment_score: 0.7,
        keywords_count: 25,
        sentiment_trend: 0.0,
        days_until_checkin: 14,
        surge_pricing: 0,
        competitor_price: 800.0,
        competitor_availability: 0.5,
        price_last_week: 750.0,
        hostel_age: 10,
        female_dorm: 0,
        common_areas: 3,
        room_area: 200,
        topic_category: 'location'
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result, data);
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to prediction service. Please ensure the backend server is running.');
    }
});

// Show loading state
function showLoading() {
    placeholder.style.display = 'none';
    resultsContent.style.display = 'none';
    errorState.style.display = 'none';
    loadingState.style.display = 'block';
}

// Display prediction results
function displayResults(result, inputData) {
    const { prediction, input_summary, timestamp } = result;
    
    // Hide loading, show results
    loadingState.style.display = 'none';
    errorState.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Update price (INR formatting)
    document.getElementById('predictedPrice').textContent = `₹${prediction.price.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    
    // Update confidence interval
    document.getElementById('confidenceMin').textContent = `₹${prediction.confidence_interval.lower.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById('confidenceMax').textContent = `₹${prediction.confidence_interval.upper.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    
    // Update confidence bar
    const range = prediction.confidence_interval.upper - prediction.confidence_interval.lower;
    const position = ((prediction.price - prediction.confidence_interval.lower) / range) * 100;
    document.getElementById('confidenceMarker').style.left = `${position}%`;
    document.getElementById('confidenceFill').style.width = '100%';
    
    // Update timestamp
    const date = new Date(timestamp);
    document.getElementById('timestamp').textContent = date.toLocaleString();
    
    // Create summary
    const summaryHTML = `
        <div class="summary-item">
            <span class="summary-label">City</span>
            <span class="summary-value">${input_summary.city}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Room Type</span>
            <span class="summary-value">${input_summary.room_type === 'private' ? 'Private Room' : 'Shared Dorm'}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Rating</span>
            <span class="summary-value">${input_summary.rating} ⭐</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Beds</span>
            <span class="summary-value">${input_summary.beds} beds</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Season</span>
            <span class="summary-value">${inputData.season.charAt(0).toUpperCase() + inputData.season.slice(1)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Occupancy</span>
            <span class="summary-value">${(inputData.occupancy_rate * 100).toFixed(0)}%</span>
        </div>
    `;
    
    document.getElementById('resultSummary').innerHTML = summaryHTML;
    
    // Add animation
    resultsContent.style.opacity = '0';
    setTimeout(() => {
        resultsContent.style.transition = 'opacity 0.5s';
        resultsContent.style.opacity = '1';
    }, 50);
    
    // Scroll to results on mobile
    if (window.innerWidth < 1024) {
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Show error state
function showError(message) {
    loadingState.style.display = 'none';
    resultsContent.style.display = 'none';
    errorState.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

// Hide error and reset
function hideError() {
    errorState.style.display = 'none';
    placeholder.style.display = 'block';
}

// Show new prediction form
function showPredictionAnother() {
    resultsContent.style.display = 'none';
    placeholder.style.display = 'block';
    
    // Scroll to form on mobile
    if (window.innerWidth < 1024) {
        form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Reset form
function resetForm() {
    form.reset();
    occupancyDisplay.textContent = '70%';
    hideError();
    
    // Reset to placeholder
    resultsContent.style.display = 'none';
    loadingState.style.display = 'none';
    errorState.style.display = 'none';
    placeholder.style.display = 'block';
}

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            console.log('✅ API is healthy and model is loaded');
        } else {
            console.warn('⚠️ API is running but model may not be loaded');
        }
    } catch (error) {
        console.error('❌ API health check failed:', error);
        console.warn('Make sure to start the backend server: python backend/app.py');
    }
}

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    
    // Add active state to nav on scroll
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
});

// Make functions globally available
window.resetForm = resetForm;
window.hideError = hideError;
window.showPredictionAnother = showPredictionAnother;
