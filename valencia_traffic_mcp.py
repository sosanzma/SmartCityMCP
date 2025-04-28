"""
Valencia Smart City MCP Server

This MCP server provides access to open data from Valencia, including traffic information,
air quality data, and weather information. It connects to the city's open data APIs and 
Open-Meteo weather API to expose the data through resources and tools.
"""

import asyncio
import httpx
import math
from mcp.server.fastmcp import FastMCP, Context, Image
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from collections import Counter

# Create the MCP server
mcp = FastMCP("Valencia Smart City Data", instructions="""
This server provides access to Valencia's urban data, currently focusing on traffic information,
air quality monitoring and weather data.
- Use resources to access raw data snapshots
- Use tools to query and filter specific information
""")

# Base URL for Valencia's open data API
BASE_URL = "https://valencia.opendatasoft.com"

# Coordinates for Valencia
VALENCIA_LATITUDE = 39.47
VALENCIA_LONGITUDE = -0.38

# Traffic state names mapping
TRAFFIC_STATE_NAMES = {
    0: "Fluido",
    1: "Denso",
    2: "Congestionado",
    3: "Cortado",
    4: "Sin datos",
    5: "Paso inferior fluido",
    6: "Paso inferior denso",
    7: "Paso inferior congestionado",
    8: "Paso inferior cortado",
    9: "Sin datos (paso inferior)"
}

# Air quality index mapping
AIR_QUALITY_RATINGS = [
    "Excelente",
    "Buena",
    "Razonablemente Buena",
    "Regular",
    "Desfavorable",
    "Muy Desfavorable",
    "Extremadamente Desfavorable"
]

# Weather code mapping
WEATHER_CODES = {
    0: "Cielo despejado",
    1: "Mayormente despejado",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Niebla",
    48: "Niebla con escarcha",
    51: "Llovizna ligera",
    53: "Llovizna moderada",
    55: "Llovizna densa",
    56: "Llovizna helada ligera",
    57: "Llovizna helada densa",
    61: "Lluvia ligera",
    63: "Lluvia moderada", 
    65: "Lluvia intensa",
    66: "Lluvia helada ligera",
    67: "Lluvia helada intensa",
    71: "Nevada ligera",
    73: "Nevada moderada",
    75: "Nevada intensa",
    77: "Granos de nieve",
    80: "Chubascos ligeros",
    81: "Chubascos moderados",
    82: "Chubascos intensos",
    85: "Chubascos de nieve ligeros",
    86: "Chubascos de nieve intensos",
    95: "Tormenta",
    96: "Tormenta con granizo ligero",
    99: "Tormenta con granizo fuerte"
}

class TrafficRecord(BaseModel):
    """Represents a traffic record for a specific road segment."""
    segment_id: str = Field(description="Unique identifier for the road segment")
    name: str = Field(description="Name of the road segment")
    state_code: int = Field(description="Numeric code representing traffic state")
    state_name: str = Field(description="Human-readable description of traffic state")
    timestamp: str = Field(description="Timestamp of the data")


class BikeStation(BaseModel):
    """Represents a Valenbisi bike rental station."""
    station_id: str = Field(description="Unique identifier for the station")
    name: str = Field(description="Name of the station")
    address: str = Field(description="Address of the station")
    available_bikes: int = Field(description="Number of available bikes")
    free_slots: int = Field(description="Number of free slots")
    total_slots: int = Field(description="Total capacity of the station")
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    timestamp: str = Field(description="Last update timestamp")
    is_open: bool = Field(description="Whether the station is operational")
    has_ticket: bool = Field(default=False, description="Whether the station has ticket functionality")


class AirQualityStation(BaseModel):
    """Represents an air quality monitoring station."""
    station_id: str = Field(description="Unique identifier for the station")
    name: str = Field(description="Name of the station")
    address: str = Field(description="Address of the station")
    zone_type: str = Field(description="Type of zone (Urban, Suburban, etc.)")
    emission_type: str = Field(description="Type of emission measured (Traffic, Background, etc.)")
    parameters: List[str] = Field(description="Parameters measured at this station")
    quality_rating: str = Field(description="Overall air quality rating")
    so2: Optional[float] = Field(description="Sulfur dioxide level (μg/m³)")
    no2: Optional[float] = Field(description="Nitrogen dioxide level (μg/m³)")
    o3: Optional[float] = Field(description="Ozone level (μg/m³)")
    co: Optional[float] = Field(description="Carbon monoxide level (mg/m³)")
    pm10: Optional[float] = Field(description="PM10 particulate matter level (μg/m³)")
    pm25: Optional[float] = Field(description="PM2.5 particulate matter level (μg/m³)")
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    timestamp: str = Field(description="Last update timestamp")


class CurrentWeather(BaseModel):
    """Represents current weather conditions in Valencia."""
    temperature: float = Field(description="Current temperature in Celsius")
    apparent_temperature: Optional[float] = Field(None, description="Apparent temperature in Celsius")
    relative_humidity: Optional[float] = Field(None, description="Relative humidity in %")
    wind_speed: Optional[float] = Field(None, description="Wind speed in km/h")
    wind_direction: Optional[int] = Field(None, description="Wind direction in degrees")
    precipitation: Optional[float] = Field(None, description="Precipitation in mm")
    weather_code: Optional[int] = Field(None, description="Weather condition code")
    weather_description: Optional[str] = Field(None, description="Human-readable weather description")
    timestamp: str = Field(description="Timestamp of the weather data")


class WeatherForecast(BaseModel):
    """Represents a weather forecast entry for a specific time."""
    time: str = Field(description="Forecast time")
    temperature: float = Field(description="Forecasted temperature in Celsius")
    apparent_temperature: Optional[float] = Field(None, description="Apparent temperature in Celsius")
    relative_humidity: Optional[float] = Field(None, description="Forecasted relative humidity in %")
    wind_speed: Optional[float] = Field(None, description="Forecasted wind speed in km/h")
    wind_direction: Optional[int] = Field(None, description="Wind direction in degrees")
    precipitation_probability: Optional[float] = Field(None, description="Probability of precipitation in %")
    precipitation: Optional[float] = Field(None, description="Precipitation amount in mm")
    weather_code: Optional[int] = Field(None, description="Forecasted weather condition code")
    weather_description: Optional[str] = Field(None, description="Human-readable weather description")


# Traffic API functions
async def fetch_traffic_data() -> List[Dict[str, Any]]:
    """Fetch traffic data from Valencia's open data API."""
    url = f"{BASE_URL}/api/explore/v2.1/catalog/datasets/estat-transit-temps-real-estado-trafico-tiempo-real/records"
    params = {"limit": 20}  # Using suggested limit
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])


async def fetch_bike_stations() -> List[Dict[str, Any]]:
    """Fetch bike station data from Valencia's open data API."""
    url = f"{BASE_URL}/api/explore/v2.1/catalog/datasets/valenbisi-disponibilitat-valenbisi-dsiponibilidad/records"
    params = {"limit": 20}  # Using suggested limit
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])


async def fetch_air_quality_data() -> List[Dict[str, Any]]:
    """Fetch air quality data from Valencia's open data API."""
    url = f"{BASE_URL}/api/explore/v2.1/catalog/datasets/estacions-contaminacio-atmosferiques-estaciones-contaminacion-atmosfericas/records"
    params = {"limit": 20}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])


# Weather API functions
async def fetch_current_weather() -> Dict[str, Any]:
    """Fetch current weather data for Valencia from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": VALENCIA_LATITUDE,
        "longitude": VALENCIA_LONGITUDE,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def fetch_weather_forecast(days: int = 3) -> Dict[str, Any]:
    """Fetch weather forecast data for Valencia from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": VALENCIA_LATITUDE,
        "longitude": VALENCIA_LONGITUDE,
        "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation_probability,precipitation,weather_code",
        "forecast_days": min(7, max(1, days))  # Limit between 1-7 days
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def fetch_historical_weather(start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch historical weather data for Valencia from Open-Meteo API."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": VALENCIA_LATITUDE,
        "longitude": VALENCIA_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


# Processors
def process_traffic_records(records: List[Dict[str, Any]]) -> List[TrafficRecord]:
    """Process traffic records into a more usable format."""
    processed_records = []
    
    for record in records:
        segment_id = record.get("idtramo", "Unknown")
        name = record.get("denominacion", "Unknown")
        state_code = record.get("estado", 4)  # Default to "Sin datos"
        state_name = TRAFFIC_STATE_NAMES.get(state_code, "Estado desconocido")
        
        processed_records.append(
            TrafficRecord(
                segment_id=str(segment_id),
                name=name,
                state_code=state_code,
                state_name=state_name,
                timestamp=datetime.now().isoformat()
            )
        )
    
    return processed_records


def process_bike_stations(records: List[Dict[str, Any]]) -> List[BikeStation]:
    """Process bike station records into a more usable format."""
    processed_stations = []
    
    for record in records:
        station_id = str(record.get("number", "Unknown"))
        address = record.get("address", "Unknown")
        is_open = record.get("open", "F") == "T"  # T for true, F for false
        available_bikes = record.get("available", 0)
        free_slots = record.get("free", 0)
        total_slots = record.get("total", 0)
        has_ticket = record.get("ticket", "F") == "T"
        updated_at = record.get("updated_at", "")
        
        # Extract coordinates from geo_point_2d
        geo_point = record.get("geo_point_2d", {})
        latitude = geo_point.get("lat", 0.0)
        longitude = geo_point.get("lon", 0.0)
        
        processed_stations.append(
            BikeStation(
                station_id=station_id,
                name=f"Valenbisi #{station_id}",
                address=address,
                available_bikes=available_bikes,
                free_slots=free_slots,
                total_slots=total_slots,
                latitude=latitude,
                longitude=longitude,
                timestamp=updated_at,
                is_open=is_open,
                has_ticket=has_ticket
            )
        )
    
    return processed_stations


def process_air_quality_stations(records: List[Dict[str, Any]]) -> List[AirQualityStation]:
    """Process air quality station records into a more usable format."""
    processed_stations = []
    
    for record in records:
        station_id = str(record.get("objectid", "Unknown"))
        name = record.get("nombre", "Unknown")
        address = record.get("direccion", "Unknown")
        zone_type = record.get("tipozona", "Unknown")
        emission_type = record.get("tipoemisio", "Unknown")
        quality_rating = record.get("calidad_am", "Unknown")
        
        # Get parameters as a list
        parameters_str = record.get("parametros", "")
        parameters = [p.strip() for p in parameters_str.split(",")] if parameters_str else []
        
        # Extract measurements
        so2 = record.get("so2")
        no2 = record.get("no2")
        o3 = record.get("o3")
        co = record.get("co")
        pm10 = record.get("pm10")
        pm25 = record.get("pm25")
        
        # Timestamp from data
        timestamp = record.get("fecha_carg", datetime.now().isoformat())
        
        # Extract coordinates from geo_point_2d
        geo_point = record.get("geo_point_2d", {})
        latitude = geo_point.get("lat", 0.0)
        longitude = geo_point.get("lon", 0.0)
        
        processed_stations.append(
            AirQualityStation(
                station_id=station_id,
                name=name,
                address=address,
                zone_type=zone_type,
                emission_type=emission_type,
                parameters=parameters,
                quality_rating=quality_rating,
                so2=so2,
                no2=no2,
                o3=o3,
                co=co,
                pm10=pm10,
                pm25=pm25,
                latitude=latitude,
                longitude=longitude,
                timestamp=timestamp
            )
        )
    
    return processed_stations


def process_current_weather(data: Dict[str, Any]) -> CurrentWeather:
    """Process current weather data into a structured format."""
    current = data.get("current", {})
    
    # Map weather code to description
    weather_code = current.get("weather_code")
    weather_description = WEATHER_CODES.get(weather_code, "Desconocido") if weather_code is not None else None
    
    return CurrentWeather(
        temperature=current.get("temperature_2m"),
        apparent_temperature=current.get("apparent_temperature"),
        relative_humidity=current.get("relative_humidity_2m"),
        wind_speed=current.get("wind_speed_10m"),
        wind_direction=current.get("wind_direction_10m"),
        precipitation=current.get("precipitation"),
        weather_code=weather_code,
        weather_description=weather_description,
        timestamp=current.get("time", datetime.now().isoformat())
    )


def process_weather_forecast(data: Dict[str, Any]) -> List[WeatherForecast]:
    """Process weather forecast data into a list of structured forecasts."""
    hourly = data.get("hourly", {})
    time_entries = hourly.get("time", [])
    
    forecasts = []
    for i, time in enumerate(time_entries):
        # Check that the index is within range to avoid errors
        if i >= len(hourly.get("temperature_2m", [])):
            continue
            
        weather_code = hourly.get("weather_code", [])[i] if i < len(hourly.get("weather_code", [])) else None
        
        forecasts.append(
            WeatherForecast(
                time=time,
                temperature=hourly.get("temperature_2m", [])[i],
                apparent_temperature=hourly.get("apparent_temperature", [])[i] if i < len(hourly.get("apparent_temperature", [])) else None,
                relative_humidity=hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
                wind_speed=hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
                wind_direction=hourly.get("wind_direction_10m", [])[i] if i < len(hourly.get("wind_direction_10m", [])) else None,
                precipitation_probability=hourly.get("precipitation_probability", [])[i] if i < len(hourly.get("precipitation_probability", [])) else None,
                precipitation=hourly.get("precipitation", [])[i] if i < len(hourly.get("precipitation", [])) else None,
                weather_code=weather_code,
                weather_description=WEATHER_CODES.get(weather_code, "Desconocido") if weather_code is not None else None
            )
        )
    
    return forecasts


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def find_closest_station(stations: List[AirQualityStation], lat: float, lon: float) -> Tuple[AirQualityStation, float]:
    """Find the closest station to a given coordinate.
    
    Args:
        stations: List of AirQualityStation objects
        lat: Latitude to search from
        lon: Longitude to search from
        
    Returns:
        Tuple of (closest station, distance in km)
    """
    if not stations:
        raise ValueError("No stations provided")
    
    closest_station = None
    min_distance = float('inf')
    
    for station in stations:
        distance = calculate_distance(lat, lon, station.latitude, station.longitude)
        if distance < min_distance:
            min_distance = distance
            closest_station = station
    
    return (closest_station, min_distance)


# Resources

@mcp.resource("valencia://traffic/raw")
async def get_raw_traffic_data() -> Dict[str, Any]:
    """Get raw traffic data from Valencia's open data API."""
    records = await fetch_traffic_data()
    return {"data": records, "timestamp": datetime.now().isoformat()}


@mcp.resource("valencia://traffic/processed")
async def get_processed_traffic_data() -> List[Dict[str, Any]]:
    """Get processed traffic data with human-readable state information."""
    records = await fetch_traffic_data()
    processed_records = process_traffic_records(records)
    return [record.model_dump() for record in processed_records]


@mcp.resource("valencia://bikes/raw")
async def get_raw_bike_station_data() -> Dict[str, Any]:
    """Get raw bike station data from Valencia's open data API."""
    records = await fetch_bike_stations()
    return {"data": records, "timestamp": datetime.now().isoformat()}


@mcp.resource("valencia://bikes/processed")
async def get_processed_bike_station_data() -> List[Dict[str, Any]]:
    """Get processed bike station data with formatted fields."""
    records = await fetch_bike_stations()
    processed_stations = process_bike_stations(records)
    return [station.model_dump() for station in processed_stations]


@mcp.resource("valencia://air-quality/raw")
async def get_raw_air_quality_data() -> Dict[str, Any]:
    """Get raw air quality data from Valencia's open data API."""
    records = await fetch_air_quality_data()
    return {"data": records, "timestamp": datetime.now().isoformat()}


@mcp.resource("valencia://air-quality/processed")
async def get_processed_air_quality_data() -> List[Dict[str, Any]]:
    """Get processed air quality data with formatted fields."""
    records = await fetch_air_quality_data()
    processed_stations = process_air_quality_stations(records)
    return [station.model_dump() for station in processed_stations]


@mcp.resource("valencia://weather/current")
async def get_current_weather_resource() -> Dict[str, Any]:
    """Get current weather conditions in Valencia."""
    data = await fetch_current_weather()
    weather = process_current_weather(data)
    return weather.model_dump()


@mcp.resource("valencia://weather/forecast")
async def get_weather_forecast_resource() -> Dict[str, Any]:
    """Get weather forecast for Valencia."""
    data = await fetch_weather_forecast()
    forecasts = process_weather_forecast(data)
    
    # Group by day for easier access
    forecast_by_day = {}
    for forecast in forecasts:
        day = forecast.time.split("T")[0]
        if day not in forecast_by_day:
            forecast_by_day[day] = []
        forecast_by_day[day].append(forecast.model_dump())
    
    return {
        "location": "Valencia, España",
        "coordinates": {
            "latitude": VALENCIA_LATITUDE,
            "longitude": VALENCIA_LONGITUDE
        },
        "forecast_by_day": forecast_by_day,
        "hourly_forecasts": [f.model_dump() for f in forecasts]
    }


@mcp.resource("valencia://weather/historical/{start_date}/{end_date}")
async def get_historical_weather_resource(start_date: str, end_date: str) -> Dict[str, Any]:
    """Get historical weather data for Valencia for the specified period."""
    return await get_historical_weather(start_date, end_date)


# Traffic Tools

@mcp.tool()
async def get_traffic_status(state_filter: Optional[List[int]] = None) -> List[str]:
    """
    Get the current traffic status for road segments, optionally filtered by state.
    
    Args:
        state_filter: Optional list of state codes to filter by. If None, returns all records.
                     State codes:
                     0: Fluido, 1: Denso, 2: Congestionado, 3: Cortado, 4: Sin datos,
                     5-9: Mismo para pasos inferiores
    
    Returns:
        List of strings describing traffic conditions for matching segments
    """
    # Para debug, intentar directamente con httpx
    url = f"{BASE_URL}/api/explore/v2.1/catalog/datasets/estat-transit-temps-real-estado-trafico-tiempo-real/records"
    params = {"limit": 20}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        raw_records = data.get("results", [])
        
        if not raw_records:
            return ["No traffic information was found in the API response."]
        
        # Procesar directamente los registros
        processed_records = []
        for record in raw_records:
            segment_id = record.get("idtramo", "Unknown")
            name = record.get("denominacion", "Unknown")
            state_code = record.get("estado", 4)  # Default to "Sin datos"
            state_name = TRAFFIC_STATE_NAMES.get(state_code, "Estado desconocido")
            
            if state_filter is None or state_code in state_filter:
                processed_records.append(f"Calle: {name}, Estado: {state_name}")
        
        if not processed_records:
            return ["No traffic information matching your criteria was found."]
        
        return processed_records


@mcp.tool()
async def get_congestion_summary() -> Dict[str, Any]:
    """
    Get a summary of current traffic congestion in Valencia.
    
    Returns:
        A dictionary with congestion statistics: counts by state and list of congested segments
    """
    records = await fetch_traffic_data()
    processed_records = process_traffic_records(records)
    
    # Count records by state
    state_counts = {}
    for state_code, state_name in TRAFFIC_STATE_NAMES.items():
        count = sum(1 for r in processed_records if r.state_code == state_code)
        state_counts[state_name] = count
    
    # Get congested segments (state codes 2, 3, 7, 8)
    congestion_codes = [2, 3, 7, 8]
    congested_segments = [
        {"name": r.name, "state": r.state_name}
        for r in processed_records
        if r.state_code in congestion_codes
    ]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "state_counts": state_counts,
        "congested_segments": congested_segments,
        "total_segments": len(processed_records),
        "congested_percentage": round(len(congested_segments) / len(processed_records) * 100, 2) if processed_records else 0
    }


@mcp.tool()
async def search_road_segment(name_query: str) -> List[Dict[str, Any]]:
    """
    Search for road segments by name.
    
    Args:
        name_query: Partial or full name of the road segment to search for
                   (case-insensitive)
    
    Returns:
        List of matching road segments with their current traffic state
    """
    records = await fetch_traffic_data()
    processed_records = process_traffic_records(records)
    
    # Case-insensitive search
    lower_query = name_query.lower()
    matching_records = [
        r.model_dump() for r in processed_records 
        if lower_query in r.name.lower()
    ]
    
    return matching_records


# Bike Tools

@mcp.tool()
async def find_available_bikes(min_bikes: int = 1, near_address: str = None) -> Dict[str, Any]:
    """
    Find bike stations with available bikes.
    
    Args:
        min_bikes: Minimum number of available bikes required (default: 1)
        near_address: Optional address or landmark to search near (case-insensitive)
    
    Returns:
        Dictionary with matching bike stations and summary information
    """
    records = await fetch_bike_stations()
    processed_stations = process_bike_stations(records)
    
    # Filter by available bikes
    available_stations = [s for s in processed_stations if s.available_bikes >= min_bikes]
    
    # Filter by address if provided
    if near_address:
        lower_address = near_address.lower()
        available_stations = [
            s for s in available_stations
            if lower_address in s.address.lower()
        ]
    
    # Prepare response
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_stations": len(processed_stations),
        "matching_stations": len(available_stations),
        "total_available_bikes": sum(s.available_bikes for s in available_stations),
        "stations": [s.model_dump() for s in available_stations]
    }
    
    return result


@mcp.tool()
async def get_bike_station_status(station_id: str = None) -> Dict[str, Any]:
    """
    Get detailed status of bike stations, optionally filtered by station ID.
    
    Args:
        station_id: Optional station ID to get specific station information
    
    Returns:
        Dictionary with bike station status information
    """
    records = await fetch_bike_stations()
    processed_stations = process_bike_stations(records)
    
    # Filter by station ID if provided
    if station_id:
        stations = [s for s in processed_stations if s.station_id == station_id]
        if not stations:
            return {
                "error": f"No station found with ID {station_id}",
                "available_stations": len(processed_stations),
                "timestamp": datetime.now().isoformat()
            }
    else:
        stations = processed_stations
    
    # Calculate summary statistics
    total_bikes = sum(s.available_bikes for s in stations)
    total_slots = sum(s.total_slots for s in stations)
    total_free = sum(s.free_slots for s in stations)
    
    # Count stations by capacity status
    full_stations = [s for s in stations if s.free_slots == 0]
    empty_stations = [s for s in stations if s.available_bikes == 0]
    optimal_stations = [
        s for s in stations 
        if s.available_bikes > 0 and s.free_slots > 0
    ]
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "total_stations": len(stations),
        "total_bikes_available": total_bikes,
        "total_free_slots": total_free,
        "total_capacity": total_slots,
        "usage_percentage": round((total_bikes / total_slots) * 100, 2) if total_slots > 0 else 0,
        "status_summary": {
            "full_stations": len(full_stations),
            "empty_stations": len(empty_stations),
            "optimal_stations": len(optimal_stations)
        },
        "stations": [s.model_dump() for s in stations]
    }
    
    return result


# Air Quality Tools

@mcp.tool()
async def get_air_quality_summary() -> Dict[str, Any]:
    """
    Get a summary of current air quality conditions in Valencia.
    
    Returns:
        Dictionary with air quality statistics and ratings across the city
    """
    records = await fetch_air_quality_data()
    stations = process_air_quality_stations(records)
    
    # Count stations by air quality rating
    quality_counts = {}
    for rating in AIR_QUALITY_RATINGS:
        count = sum(1 for s in stations if s.quality_rating == rating)
        if count > 0:  # Only include ratings that have stations
            quality_counts[rating] = count
    
    # Calculate average pollutant levels across all stations
    pollutants = {
        "so2": [],
        "no2": [],
        "o3": [],
        "co": [],
        "pm10": [],
        "pm25": []
    }
    
    for station in stations:
        if station.so2 is not None and station.so2 > 0:
            pollutants["so2"].append(station.so2)
        if station.no2 is not None and station.no2 > 0:
            pollutants["no2"].append(station.no2)
        if station.o3 is not None and station.o3 > 0:
            pollutants["o3"].append(station.o3)
        if station.co is not None and station.co > 0:
            pollutants["co"].append(station.co)
        if station.pm10 is not None and station.pm10 > 0:
            pollutants["pm10"].append(station.pm10)
        if station.pm25 is not None and station.pm25 > 0:
            pollutants["pm25"].append(station.pm25)
    
    # Calculate averages
    avg_pollutants = {}
    for key, values in pollutants.items():
        if values:
            avg_pollutants[key] = round(sum(values) / len(values), 2)
        else:
            avg_pollutants[key] = None
    
    # Find stations with highest pollutant levels
    highest_no2 = max(stations, key=lambda s: s.no2 if s.no2 is not None else -1)
    highest_pm10 = max(stations, key=lambda s: s.pm10 if s.pm10 is not None else -1)
    highest_o3 = max(stations, key=lambda s: s.o3 if s.o3 is not None else -1)
    
    high_pollution_stations = []
    for station in [highest_no2, highest_pm10, highest_o3]:
        if not any(s["station_id"] == station.station_id for s in high_pollution_stations):
            high_pollution_stations.append({
                "station_id": station.station_id, 
                "name": station.name,
                "quality_rating": station.quality_rating,
                "main_pollutants": {
                    "no2": station.no2,
                    "pm10": station.pm10,
                    "o3": station.o3
                }
            })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total_stations": len(stations),
        "quality_distribution": quality_counts,
        "average_pollutant_levels": avg_pollutants,
        "highest_pollution_stations": high_pollution_stations
    }


@mcp.tool()
async def find_nearest_air_station(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Find the air quality monitoring station nearest to specified coordinates.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
    
    Returns:
        Dictionary with information about the nearest station, its distance, and current readings
    """
    records = await fetch_air_quality_data()
    stations = process_air_quality_stations(records)
    
    if not stations:
        return {"error": "No air quality stations found in the data"}
    
    # Find the closest station
    closest_station, distance = find_closest_station(stations, latitude, longitude)
    
    if not closest_station:
        return {"error": "Could not determine closest station"}
    
    # Prepare pollutant data
    pollutants = {}
    if closest_station.so2 is not None:
        pollutants["so2"] = closest_station.so2
    if closest_station.no2 is not None:
        pollutants["no2"] = closest_station.no2
    if closest_station.o3 is not None:
        pollutants["o3"] = closest_station.o3
    if closest_station.co is not None:
        pollutants["co"] = closest_station.co
    if closest_station.pm10 is not None and closest_station.pm10 > 0:
        pollutants["pm10"] = closest_station.pm10
    if closest_station.pm25 is not None and closest_station.pm25 > 0:
        pollutants["pm25"] = closest_station.pm25
    
    return {
        "station": {
            "id": closest_station.station_id,
            "name": closest_station.name,
            "address": closest_station.address,
            "zone_type": closest_station.zone_type,
            "emission_type": closest_station.emission_type,
            "quality_rating": closest_station.quality_rating,
            "coordinates": {
                "latitude": closest_station.latitude,
                "longitude": closest_station.longitude
            }
        },
        "distance_km": round(distance, 2),
        "pollutant_levels": pollutants,
        "timestamp": closest_station.timestamp
    }


@mcp.tool()
async def get_station_data(station_name: str) -> Dict[str, Any]:
    """
    Get detailed data for a specific air quality monitoring station.
    
    Args:
        station_name: Name of the station (case-insensitive partial match)
    
    Returns:
        Dictionary with detailed information about matching stations
    """
    records = await fetch_air_quality_data()
    stations = process_air_quality_stations(records)
    
    # Find stations matching the name (case-insensitive)
    lower_name = station_name.lower()
    matching_stations = [
        s for s in stations
        if lower_name in s.name.lower() or lower_name in s.address.lower()
    ]
    
    if not matching_stations:
        return {
            "error": f"No stations found matching '{station_name}'",
            "available_stations": [s.name for s in stations]
        }
    
    # Convert to dict for response
    result = {
        "count": len(matching_stations),
        "stations": []
    }
    
    for station in matching_stations:
        pollutants = {}
        if station.so2 is not None:
            pollutants["so2"] = {"value": station.so2, "unit": "μg/m³"}
        if station.no2 is not None:
            pollutants["no2"] = {"value": station.no2, "unit": "μg/m³"}
        if station.o3 is not None:
            pollutants["o3"] = {"value": station.o3, "unit": "μg/m³"}
        if station.co is not None:
            pollutants["co"] = {"value": station.co, "unit": "mg/m³"}
        if station.pm10 is not None and station.pm10 > 0:
            pollutants["pm10"] = {"value": station.pm10, "unit": "μg/m³"}
        if station.pm25 is not None and station.pm25 > 0:
            pollutants["pm25"] = {"value": station.pm25, "unit": "μg/m³"}
        
        result["stations"].append({
            "id": station.station_id,
            "name": station.name,
            "address": station.address,
            "zone_type": station.zone_type,
            "emission_type": station.emission_type,
            "quality_rating": station.quality_rating,
            "parameters_measured": station.parameters,
            "pollutant_levels": pollutants,
            "coordinates": {
                "latitude": station.latitude,
                "longitude": station.longitude
            },
            "timestamp": station.timestamp
        })
    
    return result


@mcp.tool()
async def get_pollutant_levels(pollutant: str) -> Dict[str, Any]:
    """
    Get current levels of a specific pollutant across all monitoring stations.
    
    Args:
        pollutant: Type of pollutant to check (so2, no2, o3, co, pm10, pm25)
    
    Returns:
        Dictionary with levels of the specified pollutant at all measuring stations
    """
    valid_pollutants = ["so2", "no2", "o3", "co", "pm10", "pm25"]
    pollutant = pollutant.lower()
    
    if pollutant not in valid_pollutants:
        return {
            "error": f"Invalid pollutant. Valid options are: {', '.join(valid_pollutants)}",
            "valid_pollutants": valid_pollutants
        }
    
    records = await fetch_air_quality_data()
    stations = process_air_quality_stations(records)
    
    # Find stations measuring this pollutant
    measuring_stations = []
    for station in stations:
        # Get the pollutant value using getattr
        value = getattr(station, pollutant, None)
        if value is not None and value > 0:  # Only include stations with valid readings
            measuring_stations.append({
                "id": station.station_id,
                "name": station.name,
                "address": station.address,
                "zone_type": station.zone_type,
                "emission_type": station.emission_type,
                "value": value,
                "quality_rating": station.quality_rating,
                "coordinates": {
                    "latitude": station.latitude,
                    "longitude": station.longitude
                },
                "timestamp": station.timestamp
            })
    
    # Sort stations by pollutant level (highest first)
    measuring_stations.sort(key=lambda s: s["value"], reverse=True)
    
    # Calculate statistics
    if measuring_stations:
        values = [s["value"] for s in measuring_stations]
        avg_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)
    else:
        avg_value = max_value = min_value = 0
    
    result = {
        "pollutant": pollutant,
        "stations_measuring": len(measuring_stations),
        "statistics": {
            "average": round(avg_value, 2),
            "maximum": max_value,
            "minimum": min_value
        },
        "unit": "μg/m³" if pollutant != "co" else "mg/m³",
        "stations": measuring_stations
    }
    
    return result


@mcp.tool()
async def get_air_quality_map() -> Dict[str, Any]:
    """
    Get data for creating an air quality map of Valencia.
    
    Returns:
        Dictionary with geospatial data for all air quality monitoring stations
    """
    records = await fetch_air_quality_data()
    stations = process_air_quality_stations(records)
    
    # Prepare geospatial data
    geospatial_data = []
    for station in stations:
        # Only include stations with valid coordinates
        if station.latitude != 0 and station.longitude != 0:
            pollutants = {}
            if station.so2 is not None:
                pollutants["so2"] = station.so2
            if station.no2 is not None:
                pollutants["no2"] = station.no2
            if station.o3 is not None:
                pollutants["o3"] = station.o3
            if station.co is not None:
                pollutants["co"] = station.co
            if station.pm10 is not None and station.pm10 > 0:
                pollutants["pm10"] = station.pm10
            if station.pm25 is not None and station.pm25 > 0:
                pollutants["pm25"] = station.pm25
            
            geospatial_data.append({
                "id": station.station_id,
                "name": station.name,
                "quality_rating": station.quality_rating,
                "pollutants": pollutants,
                "type": station.zone_type,
                "coordinates": {
                    "latitude": station.latitude,
                    "longitude": station.longitude
                }
            })
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "center": {
            "latitude": 39.4699,  # Valencia city center approximate coordinates
            "longitude": -0.3763
        },
        "stations": geospatial_data
    }
    
    return result


# Weather Tools

@mcp.tool()
async def get_current_weather() -> Dict[str, Any]:
    """
    Get current weather conditions in Valencia.
    
    Returns:
        Dictionary with current weather information including temperature,
        humidity, wind speed, and weather description.
    """
    data = await fetch_current_weather()
    weather = process_current_weather(data)
    return weather.model_dump()


@mcp.tool()
async def get_weather_forecast(days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for Valencia.
    
    Args:
        days: Number of days to forecast (1-7, default: 3)
    
    Returns:
        Dictionary with hourly weather forecasts for the requested period.
    """
    # Validate the range of days
    days = min(7, max(1, days))
    
    data = await fetch_weather_forecast(days)
    forecasts = process_weather_forecast(data)
    
    # Group by day for easier analysis
    forecast_by_day = {}
    for forecast in forecasts:
        day = forecast.time.split("T")[0]
        if day not in forecast_by_day:
            forecast_by_day[day] = []
        forecast_by_day[day].append(forecast.model_dump())
    
    return {
        "generated_at": datetime.now().isoformat(),
        "location": "Valencia, España",
        "coordinates": {
            "latitude": VALENCIA_LATITUDE,
            "longitude": VALENCIA_LONGITUDE
        },
        "days": days,
        "forecast_by_day": forecast_by_day,
        "hourly_forecasts": [f.model_dump() for f in forecasts]
    }


@mcp.tool()
async def get_historical_weather(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Get historical weather data for Valencia.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (must be after start_date)
    
    Returns:
        Historical weather data for the specified period.
    """
    # Validate dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if end < start:
            return {"error": "La fecha final debe ser posterior a la fecha inicial"}
            
        # Limit the range to avoid very large requests
        if (end - start).days > 30:
            return {"error": "El rango de fechas no puede exceder los 30 días"}
            
    except ValueError:
        return {"error": "Formato de fecha inválido. Use YYYY-MM-DD"}
    
    data = await fetch_historical_weather(start_date, end_date)
    
    # Process historical data into a consistent format
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    
    result = {
        "generated_at": datetime.now().isoformat(),
        "location": "Valencia, España",
        "coordinates": {
            "latitude": VALENCIA_LATITUDE,
            "longitude": VALENCIA_LONGITUDE
        },
        "start_date": start_date,
        "end_date": end_date,
        "data_points": len(times),
        "hourly_data": []
    }
    
    for i, time in enumerate(times):
        if i >= len(hourly.get("temperature_2m", [])):
            continue
            
        result["hourly_data"].append({
            "time": time,
            "temperature": hourly.get("temperature_2m", [])[i],
            "relative_humidity": hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
            "wind_speed": hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
            "precipitation": hourly.get("precipitation", [])[i] if i < len(hourly.get("precipitation", [])) else None
        })
    
    return result


@mcp.tool()
async def get_daily_weather_summary(day_offset: int = 0) -> Dict[str, Any]:
    """
    Get a daily weather summary for Valencia.
    
    Args:
        day_offset: Day offset (0=today, 1=tomorrow, etc., maximum 6)
    
    Returns:
        Daily weather summary with min/max/avg values.
    """
    # Validate the day offset
    day_offset = min(6, max(0, day_offset))
    
    # Calculate the date
    target_date = datetime.now().date() + timedelta(days=day_offset)
    date_str = target_date.isoformat()
    
    # Get forecast
    data = await fetch_weather_forecast(day_offset + 1)
    forecasts = process_weather_forecast(data)
    
    # Filter forecasts for the target day
    day_forecasts = [f for f in forecasts if f.time.startswith(date_str)]
    
    if not day_forecasts:
        return {
            "error": f"No se encontraron datos para la fecha {date_str}"
        }
    
    # Calculate min/max/avg values
    temperatures = [f.temperature for f in day_forecasts if f.temperature is not None]
    humidities = [f.relative_humidity for f in day_forecasts if f.relative_humidity is not None]
    wind_speeds = [f.wind_speed for f in day_forecasts if f.wind_speed is not None]
    
    # Check if weather codes are available
    weather_codes = [f.weather_code for f in day_forecasts if f.weather_code is not None]
    
    # Find the most common code (mode)
    most_common_code = None
    if weather_codes:
        most_common_code = Counter(weather_codes).most_common(1)[0][0]
    
    # Check if precipitation probabilities are available
    precip_probs = [f.precipitation_probability for f in day_forecasts if f.precipitation_probability is not None]
    max_precip_prob = max(precip_probs) if precip_probs else None
    
    # Get day name in Spanish
    day_name = target_date.strftime("%A")
    if day_name == "Monday":
        day_name = "Lunes"
    elif day_name == "Tuesday":
        day_name = "Martes"
    elif day_name == "Wednesday":
        day_name = "Miércoles"
    elif day_name == "Thursday":
        day_name = "Jueves"
    elif day_name == "Friday":
        day_name = "Viernes"
    elif day_name == "Saturday":
        day_name = "Sábado"
    elif day_name == "Sunday":
        day_name = "Domingo"
    
    return {
        "date": date_str,
        "day_name": day_name,
        "day_type": "Hoy" if day_offset == 0 else "Mañana" if day_offset == 1 else f"Dentro de {day_offset} días",
        "temperature": {
            "min": min(temperatures) if temperatures else None,
            "max": max(temperatures) if temperatures else None,
            "avg": sum(temperatures) / len(temperatures) if temperatures else None
        },
        "humidity": {
            "min": min(humidities) if humidities else None,
            "max": max(humidities) if humidities else None,
            "avg": sum(humidities) / len(humidities) if humidities else None
        },
        "wind_speed": {
            "min": min(wind_speeds) if wind_speeds else None,
            "max": max(wind_speeds) if wind_speeds else None,
            "avg": sum(wind_speeds) / len(wind_speeds) if wind_speeds else None
        },
        "precipitation": {
            "max_probability": max_precip_prob
        },
        "predominant_weather": {
            "code": most_common_code,
            "description": WEATHER_CODES.get(most_common_code, "Desconocido") if most_common_code is not None else None
        },
        "hourly_data": [f.model_dump() for f in day_forecasts]
    }


# Prompts

@mcp.prompt("traffic_report")
def traffic_report_prompt() -> str:
    """Create a comprehensive traffic report for Valencia based on current data."""
    return """Please analyze the current traffic data from Valencia and create a comprehensive report covering:

1. Overall traffic situation
2. Areas with congestion
3. Any road closures or incidents
4. Comparison to normal conditions for this time (if historical data is available)

Use the traffic data tools to gather the necessary information for your analysis.
"""


@mcp.prompt("navigation_advice")
def navigation_advice_prompt(destination: str) -> str:
    """Provide navigation advice for reaching a specific destination in Valencia."""
    return f"""Based on the current traffic conditions in Valencia, please provide advice on reaching {destination}.

1. Search for roads leading to or near {destination}
2. Check their current traffic status
3. Suggest the best route considering traffic conditions
4. Mention any areas to avoid due to congestion or closures

Use the search_road_segment tool to find relevant roads and check their status.
"""


@mcp.prompt("bike_availability")
def bike_availability_prompt(location: str = None) -> str:
    """Create a prompt for checking bike availability near a specific location."""
    if location:
        return f"""Please analyze the current Valenbisi bike sharing availability near {location} and provide a detailed report covering:

1. Number of stations with available bikes near {location}
2. The stations with the most available bikes
3. Recommendations for where to find and return bikes
4. Any completely full or empty stations to avoid

Use the find_available_bikes tool with the parameter near_address="{location}" to gather this information.
"""
    else:
        return """Please analyze the current Valenbisi bike sharing availability in Valencia and provide a detailed report covering:

1. Overall bike availability across the city
2. Areas with high and low availability
3. Recommendations for where to find and return bikes
4. Any stations that are completely full or empty

Use the get_bike_station_status tool to gather this information.
"""


@mcp.prompt("air_quality_report")
def air_quality_report_prompt() -> str:
    """Create a comprehensive prompt for analyzing air quality in Valencia."""
    return """Please create a comprehensive air quality report for Valencia based on current monitoring data. Your report should include:

1. Overall air quality assessment
   - Current air quality ratings across the city
   - Areas with the best and worst air quality
   - Main pollutants of concern

2. Specific pollutant analysis
   - NO2 levels (traffic-related pollution)
   - Particulate matter (PM10 and PM2.5)
   - Ozone levels
   - Other notable pollutants

3. Health recommendations
   - Advice for sensitive groups (elderly, children, those with respiratory conditions)
   - Activities that should be limited if any
   - Best times/places for outdoor activities

Use the get_air_quality_summary tool and other air quality tools to gather the necessary data.
"""


@mcp.prompt("urban_mobility_report")
def urban_mobility_report_prompt() -> str:
    """Create a comprehensive prompt for analyzing urban mobility options."""
    return """Please create a comprehensive urban mobility report for Valencia covering traffic conditions, bike sharing availability, and air quality. Your report should include:

1. Overall traffic situation in Valencia
   - Major congestion points
   - Road closures or incidents
   - Recommendations for drivers

2. Valenbisi bike sharing status
   - Overall availability of bikes
   - Areas with high and low availability
   - Recommendations for users

3. Air quality considerations
   - Current air quality levels across the city
   - Areas with poor air quality to avoid
   - Relationship between traffic and air pollution

4. Integrated mobility recommendations
   - Suggestions for multimodal travel options
   - Key transfer points between transport modes
   - Time-saving and health-conscious strategies for urban travel

Use the get_congestion_summary tool for traffic data, get_bike_station_status tool for bike information, and get_air_quality_summary for environmental data.
"""


@mcp.prompt("weather_forecast")
def weather_forecast_prompt(days: int = 3) -> str:
    """Create a prompt for analyzing weather forecast in Valencia."""
    return f"""Por favor, analiza el pronóstico del tiempo en Valencia para los próximos {days} días:

1. Condiciones generales
   - Patrones de temperatura
   - Expectativas de precipitación
   - Condiciones de viento
   
2. Resumen día a día
   - Temperatura mínima y máxima
   - Probabilidad de lluvia
   - Recomendaciones generales

3. Aspectos destacados
   - Días con mejor clima
   - Alertas de condiciones adversas (si las hay)
   - Tendencias generales

Utiliza la herramienta get_weather_forecast con days={days} para obtener los datos del pronóstico.
"""


if __name__ == "__main__":
    mcp.run()
