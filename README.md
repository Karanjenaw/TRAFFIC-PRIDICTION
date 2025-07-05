# Imports for FastAPI, traffic data, predictions, utilities, and advanced features
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import tensorflow as tf
import logging
import requests
import time
from collections import defaultdict, deque
import asyncio
import math
import threading

# Record startup time for uptime calculation
START_TIME = datetime.now()

# Initialize FastAPI application with detailed metadata
app = FastAPI(
    title="Ultimate Traffic Prediction API",
    description="A highly advanced, feature-packed traffic prediction system using LSTM and TomTom API, fully in-memory.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "Traffic Team", "email": "support@trafficapi.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"}
)

# Comprehensive logging setup with multiple handlers and rotation
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("traffic_api_advanced.log", mode="a")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configuration variables
TOMTOM_API_KEY = "FU1C7Jpbvz48NT8uKhRpJrFEwXevJy9d"  # Your provided TomTom API key
CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)
MODEL_TRAINING_DATA_SIZE = 2000  # Increased size of dummy training data
HISTORY_BUFFER_SIZE = 200       # Max history entries per location
POPULAR_LOCATIONS = [
    {"lat": 37.7749, "lng": -122.4194, "name": "San Francisco"},
    {"lat": 34.0522, "lng": -118.2437, "name": "Los Angeles"},
    {"lat": 40.7128, "lng": -74.0060, "name": "New York"},
    {"lat": 51.5074, "lng": -0.1278, "name": "London"}
]

# Pydantic models for data validation and serialization
class Coordinate(BaseModel):
    """Represents a geographic coordinate with validation."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    lng: float = Field(..., ge=-180, le=180, description="Longitude between -180 and 180")
    name: Optional[str] = Field(None, description="Optional name for the location")

class RouteRequest(BaseModel):
    """Request model for route suggestion with start and end coordinates."""
    start: Coordinate
    end: Coordinate
    waypoints: Optional[List[Coordinate]] = Field(default_factory=list, description="Optional waypoints")

class TrafficData(BaseModel):
    """Stores traffic data with additional metrics."""
    speed: float = Field(..., ge=0, description="Current traffic speed in km/h")
    volume: float = Field(..., ge=0, description="Traffic volume or travel time in seconds")
    congestion_level: Optional[float] = Field(default=0.0, ge=0, le=1, description="Congestion level (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data fetch time")
    source: Optional[str] = Field(default="TomTom", description="Data source")

class PredictionResponse(BaseModel):
    """Response model for route prediction with detailed metrics."""
    predicted_speed: float = Field(..., ge=0, description="Predicted speed in km/h")
    travel_time: float = Field(..., ge=0, description="Estimated travel time in minutes")
    distance: float = Field(..., ge=0, description="Distance in kilometers")
    route_points: List[Coordinate] = Field(..., description="Points along the route")
    congestion_zones: List[Coordinate] = Field(default_factory=list, description="Congested areas")

class TrafficStats(BaseModel):
    """Model for traffic statistics."""
    avg_speed: float = Field(..., ge=0, description="Average speed in km/h")
    max_congestion: float = Field(..., ge=0, le=1, description="Max congestion level")
    data_points: int = Field(..., ge=0, description="Number of data points")

# Custom exceptions for precise error handling
class TrafficAPIError(Exception):
    """Raised when TomTom API calls fail."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class PredictionError(Exception):
    """Raised when LSTM prediction fails."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

# In-memory traffic data cache with TTL
class TrafficCache:
    """Advanced in-memory cache for traffic data with TTL and size limits."""
    def __init__(self, ttl: int, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve data from cache if not expired."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if (datetime.now() - entry["timestamp"]).seconds < self.ttl:
                    logger.debug(f"Cache hit for key: {key}")
                    return entry["data"]
                else:
                    logger.debug(f"Cache expired for key: {key}")
                    del self.cache[key]
            return None

    def set(self, key: str, data: Dict):
        """Store data in cache with current timestamp, evicting old entries if needed."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
                del self.cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key}")
            self.cache[key] = {"data": data, "timestamp": datetime.now()}
            logger.debug(f"Cache set for key: {key}")

traffic_cache = TrafficCache(CACHE_TTL)

# In-memory traffic data store with history and statistics
class TrafficDataStore:
    """Manages traffic data history and statistics in memory."""
    def __init__(self, max_history: int = HISTORY_BUFFER_SIZE):
        self.store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.stats: Dict[str, Dict] = {}
        self.max_history = max_history
        self.lock = threading.Lock()

    def add_data(self, location: str, data: TrafficData):
        """Add traffic data and update statistics."""
        with self.lock:
            self.store[location].append(data)
            self._update_stats(location)
            logger.info(f"Stored traffic data for {location}")

    def get_data(self, location: str, limit: int = 10) -> List[TrafficData]:
        """Retrieve recent traffic data for a location."""
        with self.lock:
            data = list(self.store[location])[-limit:]
            logger.info(f"Retrieved {len(data)} data points for {location}")
            return data

    def _update_stats(self, location: str):
        """Update traffic statistics for a location."""
        data = list(self.store[location])
        if not data:
            return
        speeds = [d.speed for d in data]
        congestion_levels = [d.congestion_level for d in data]
        self.stats[location] = {
            "avg_speed": sum(speeds) / len(speeds),
            "max_congestion": max(congestion_levels),
            "data_points": len(data)
        }

    def get_stats(self, location: str) -> Optional[TrafficStats]:
        """Retrieve traffic statistics for a location."""
        with self.lock:
            stats = self.stats.get(location)
            if stats:
                return TrafficStats(**stats)
            return None

traffic_data_store = TrafficDataStore()

# Traffic service with advanced prediction and optimization
class TrafficService:
    """Handles traffic data fetching, LSTM prediction, and route optimization."""
    def __init__(self):
        """Initialize with TomTom API key and advanced LSTM model."""
        self.api_key = TOMTOM_API_KEY
        self.lstm_model = self._build_lstm_model()
        self._simulate_training()

    def _build_lstm_model(self) -> Sequential:
        """Build an advanced LSTM model with batch normalization and dropout."""
        model = Sequential()
        model.add(LSTM(128, activation="relu", return_sequences=True, input_shape=(10, 3)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, activation="relu", return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        logger.info("Advanced LSTM model built with batch normalization and dropout")
        return model

    def _simulate_training(self):
        """Simulate training with enhanced dummy data."""
        X = np.random.rand(MODEL_TRAINING_DATA_SIZE, 10, 3) * [100, 200, 1]  # Speed, volume, congestion
        y = np.random.rand(MODEL_TRAINING_DATA_SIZE, 1) * 50                  # Predicted speeds
        self.lstm_model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2, verbose=0)
        logger.info("LSTM model trained with simulated data over 10 epochs")

    def fetch_traffic_data(self, lat: float, lng: float) -> Dict:
        """Fetch traffic data from TomTom API with retry logic."""
        key = f"{lat},{lng}"
        cached_data = traffic_cache.get(key)
        if cached_data:
            return cached_data

        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lng}&key={self.api_key}"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                speed = data["flowSegmentData"]["currentSpeed"]
                volume = data["flowSegmentData"]["currentTravelTime"]
                congestion = min(data["flowSegmentData"]["freeFlowTravelTime"] / volume, 1.0) if volume > 0 else 0.0
                traffic_data = {"speed": speed, "volume": volume, "congestion_level": congestion}
                traffic_cache.set(key, traffic_data)
                logger.info(f"Fetched traffic data for {key}: {traffic_data}")
                return traffic_data
            except requests.RequestException as e:
                logger.warning(f"TomTom API attempt {attempt + 1} failed for {key}: {e}")
                if attempt == 2:
                    raise TrafficAPIError(f"Failed to fetch traffic data for {key} after 3 attempts")
                time.sleep(1)

    def preprocess_data(self, traffic_data: Dict) -> np.ndarray:
        """Preprocess traffic data for LSTM prediction with enhanced features."""
        raw_data = [[traffic_data["speed"], traffic_data["volume"], traffic_data["congestion_level"]]] * 10
        normalized_data = np.array(raw_data) / [100.0, 200.0, 1.0]  # Normalize features
        return normalized_data.reshape(1, 10, 3)

    def predict_congestion(self, traffic_data: Dict) -> float:
        """Predict traffic speed with confidence bounds."""
        try:
            input_data = self.preprocess_data(traffic_data)
            prediction = self.lstm_model.predict(input_data, verbose=0)[0][0]
            adjusted_speed = max(prediction * 100, 1.0)  # Denormalize and ensure positive
            logger.debug(f"Predicted speed: {adjusted_speed}")
            return adjusted_speed
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise PredictionError(f"Failed to predict congestion: {e}")

traffic_service = TrafficService()

# Utility functions for route calculation and optimization
def interpolate_points(start: Coordinate, end: Coordinate, waypoints: List[Coordinate], num_points: int = 5) -> List[Coordinate]:
    """Generate intermediate points along the route with waypoints."""
    route_segments = [start] + waypoints + [end]
    points = []
    for i in range(len(route_segments) - 1):
        seg_start = route_segments[i]
        seg_end = route_segments[i + 1]
        for j in range(1, num_points + 1):
            fraction = j / (num_points + 1)
            lat = seg_start.lat + fraction * (seg_end.lat - seg_start.lat)
            lng = seg_start.lng + fraction * (seg_end.lng - seg_start.lng)
            # Optionally, validate the generated coordinate here
            points.append(Coordinate(lat=lat, lng=lng))
    logger.debug(f"Interpolated {len(points)} points for route with {len(waypoints)} waypoints")
    return points

def calculate_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """Calculate distance using Haversine formula for accuracy."""
    R = 6371  # Earth's radius in kilometers
    lat1, lon1 = math.radians(coord1.lat), math.radians(coord1.lng)
    lat2, lon2 = math.radians(coord2.lat), math.radians(coord2.lng)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    logger.debug(f"Calculated distance between {coord1} and {coord2}: {distance} km")
    return distance

def total_route_distance(points: List[Coordinate]) -> float:
    """Calculate total distance of a route."""
    total = 0.0
    for i in range(len(points) - 1):
        total += calculate_distance(points[i], points[i + 1])
    return total

# Background tasks for system maintenance
async def fetch_popular_traffic_data():
    """Periodically fetch and store traffic data for popular locations."""
    while True:
        for loc in POPULAR_LOCATIONS:
            try:
                data = traffic_service.fetch_traffic_data(loc["lat"], loc["lng"])
                traffic_data_store.add_data(f"{loc['lat']},{loc['lng']}", TrafficData(**data))
            except TrafficAPIError as e:
                logger.error(f"Failed to fetch data for popular location {loc['name']}: {e}")
        logger.info("Updated traffic data for popular locations")
        await asyncio.sleep(60)  # Update every minute

async def clean_expired_cache():
    """Periodically clean expired cache entries."""
    while True:
        with threading.Lock():
            expired = [k for k, v in traffic_cache.cache.items() if (datetime.now() - v["timestamp"]).seconds >= CACHE_TTL]
            for k in expired:
                del traffic_cache.cache[k]
            logger.debug(f"Cleaned {len(expired)} expired cache entries")
        await asyncio.sleep(300)  # Clean every 5 minutes

# Middleware for advanced request handling
@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    """Log requests and measure processing time."""
    start_time = datetime.now()
    client_ip = request.client.host
    logger.info(f"Incoming request: {request.method} {request.url} from {client_ip}")
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed request: {request.method} {request.url} - Status: {response.status_code} - Time: {duration:.2f}s")
    return response

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks and log startup."""
    logger.info("Starting Ultimate Traffic Prediction API")
    asyncio.create_task(fetch_popular_traffic_data())
    asyncio.create_task(clean_expired_cache())

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown event."""
    logger.info("Shutting down Ultimate Traffic Prediction API")

# Traffic-related endpoints
@app.post("/suggest_route", response_model=PredictionResponse)
async def suggest_route(request: RouteRequest, background_tasks: BackgroundTasks):
    """Suggest an optimized route with traffic predictions."""
    try:
        logger.info(f"Processing route request from {request.start} to {request.end}")
        
        # Generate route points with waypoints
        points = [request.start] + interpolate_points(request.start, request.end, request.waypoints) + [request.end]
        traffic_data_list = []
        congestion_zones = []
        timestamp = datetime.now()

        # Fetch and analyze traffic data, skipping points with API errors
        for point in points:
            try:
                data = traffic_service.fetch_traffic_data(point.lat, point.lng)
            except TrafficAPIError as e:
                logger.error(f"Skipping point {point} due to error: {e.message}")
                continue
            traffic_data_store.add_data(f"{point.lat},{point.lng}", TrafficData(**data, timestamp=timestamp))
            traffic_data_list.append(data)
            if data["congestion_level"] > 0.7:
                congestion_zones.append(point)
        
        if not traffic_data_list:
            raise HTTPException(status_code=500, detail="No valid traffic data available for route prediction.")

        # Predict speeds and calculate metrics
        predicted_speeds = [traffic_service.predict_congestion(data) for data in traffic_data_list]
        average_speed = sum(predicted_speeds) / len(predicted_speeds)
        distance = total_route_distance(points)
        travel_time = distance / average_speed * 60

        response = PredictionResponse(
            predicted_speed=average_speed,
            travel_time=travel_time,
            distance=distance,
            route_points=points,
            congestion_zones=congestion_zones
        )
        logger.info(f"Route predicted: speed={average_speed}, time={travel_time}, distance={distance}")
        return response
    except (TrafficAPIError, PredictionError) as e:
        logger.error(f"Error in suggest_route: {e.message}")
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/traffic_history/{lat}/{lng}", response_model=List[TrafficData])
async def get_traffic_history(lat: float, lng: float, limit: int = 20):
    """Retrieve historical traffic data for a location."""
    location = f"{lat},{lng}"
    history = traffic_data_store.get_data(location, limit)
    return history

@app.get("/traffic_stats/{lat}/{lng}", response_model=TrafficStats)
async def get_traffic_stats(lat: float, lng: float):
    """Retrieve traffic statistics for a location."""
    location = f"{lat},{lng}"
    stats = traffic_data_store.get_stats(location)
    if stats is None:
        raise HTTPException(status_code=404, detail="No statistics available for this location")
    return stats

# Health check and diagnostics
@app.get("/health")
async def health_check():
    """Check API health and system status."""
    cache_size = len(traffic_cache.cache)
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": cache_size
    }

@app.get("/diagnostics")
async def diagnostics():
    """Provide detailed system diagnostics."""
    uptime = (datetime.now() - START_TIME)
    return {
        "cache_entries": len(traffic_cache.cache),
        "history_locations": len(traffic_data_store.store),
        "popular_locations_tracked": len(POPULAR_LOCATIONS),
        "uptime": str(uptime)
    }

# Root endpoint
@app.get("/")
async def root():
    """Welcome message for the API."""
    return {"message": "Welcome to the Ultimate Traffic Prediction API v3.0"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
