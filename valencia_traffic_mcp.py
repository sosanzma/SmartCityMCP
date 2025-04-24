"""
Valencia Smart City MCP Server

This MCP server provides access to open data from Valencia, starting with traffic information.
It connects to the city's open data APIs and exposes the data through resources and tools.
"""

import asyncio
import httpx
from mcp.server.fastmcp import FastMCP, Context, Image
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Create the MCP server
mcp = FastMCP("Valencia Smart City Data", instructions="""
This server provides access to Valencia's urban data, currently focusing on traffic information.
- Use resources to access raw data snapshots
- Use tools to query and filter specific information
""")

# Base URL for Valencia's open data API
BASE_URL = "https://valencia.opendatasoft.com"

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


async def fetch_traffic_data() -> List[Dict[str, Any]]:
    """Fetch traffic data from Valencia's open data API."""
    url = f"{BASE_URL}/api/explore/v2.1/catalog/datasets/estat-transit-temps-real-estado-trafico-tiempo-real/records"
    params = {"limit": 20}  # Using suggested limit
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("records", [])


async def fetch_bike_stations() -> List[Dict[str, Any]]:
    """Fetch bike station data from Valencia's open data API."""
    url = f"{BASE_URL}/api/explore/v2.1/catalog/datasets/valenbisi-disponibilitat-valenbisi-dsiponibilidad/records"
    params = {"limit": 20}  # Using suggested limit
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])


def process_traffic_records(records: List[Dict[str, Any]]) -> List[TrafficRecord]:
    """Process traffic records into a more usable format."""
    processed_records = []
    
    for record in records:
        fields = record.get("record", {}).get("fields", {})
        segment_id = fields.get("idtramo", "Unknown")
        name = fields.get("denominacion", "Unknown")
        state_code = fields.get("estado", 4)  # Default to "Sin datos"
        state_name = TRAFFIC_STATE_NAMES.get(state_code, "Estado desconocido")
        
        processed_records.append(
            TrafficRecord(
                segment_id=segment_id,
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


# Tools

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
    records = await fetch_traffic_data()
    processed_records = process_traffic_records(records)
    
    if state_filter:
        filtered_records = [r for r in processed_records if r.state_code in state_filter]
    else:
        filtered_records = processed_records
    
    if not filtered_records:
        return ["No traffic information matching your criteria was found."]
    
    return [f"Calle: {r.name}, Estado: {r.state_name}" for r in filtered_records]


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


@mcp.prompt("urban_mobility_report")
def urban_mobility_report_prompt() -> str:
    """Create a comprehensive prompt for analyzing urban mobility options."""
    return """Please create a comprehensive urban mobility report for Valencia covering both traffic conditions and bike sharing availability. Your report should include:

1. Overall traffic situation in Valencia
   - Major congestion points
   - Road closures or incidents
   - Recommendations for drivers

2. Valenbisi bike sharing status
   - Overall availability of bikes
   - Areas with high and low availability
   - Recommendations for users

3. Integrated mobility recommendations
   - Suggestions for multimodal travel options
   - Key transfer points between transport modes
   - Time-saving strategies for urban travel

Use the get_congestion_summary tool for traffic data and get_bike_station_status tool for bike information.
"""


if __name__ == "__main__":
    mcp.run()
