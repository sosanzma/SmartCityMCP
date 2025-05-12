# Valencia Smart City MCP Server

MCP server providing real-time traffic, bike-sharing, air quality and weather data from Valencia, Spain for Claude and other LLMs.

## Features

- Real-time traffic conditions across Valencia
- Valenbisi bike station availability
- Air quality monitoring data from city stations
- Weather data from Open-Meteo API (current, forecast, and historical)
- Traffic congestion analysis
- Search capabilities for specific roads, bike stations, or air quality information

## Demo

Check out a live demos of this MCP in action with Claude Desktop :

Air quality:
[Claude MCP Demo Chat Air quality](https://claude.ai/share/1425bcc9-c7bd-4b34-9568-6e16257b494b)

Bike-sharing with Valenbisi : 
[Claude MCP Demo Chat Valenbisi](https://claude.ai/share/aa53377b-e850-4977-acf3-bbc6d7833a0b)

Traffic :
[Claude MCP Demo Chat Traffic](https://claude.ai/share/f7408546-a96d-4851-95a5-a4623358b15d)

Weather :
[Claude MCP Demo Chat Weather ](https://claude.ai/share/2d5cf806-7fd3-4d98-969e-81f4b1507021)

This shared conversation demonstrates how Claude can access and analyze real-time Valencia urban data through the MCP.

## Requirements

- Python 3.10+
- Claude Desktop or other MCP-compatible client
- Internet connection

## Installation

```bash
# Clone the repository
git clone https://github.com/sosanzma/SmartCityMCP.git
cd SmartCityMCP

# Install dependencies
pip install -r requirements.txt
```

## Tools

The server exposes these tools:

### Traffic Tools

- **get_traffic_status**
  - Get current traffic status for road segments
  - Input: `state_filter` (optional, integer[]) - Filter by traffic state codes
  - Returns: List of road segments with their traffic states

- **get_congestion_summary**
  - Get summary of current traffic congestion in Valencia
  - Returns: Statistics including congested segments and percentages

- **search_road_segment**
  - Search for specific road segments by name
  - Input: `name_query` (string) - Case-insensitive search term
  - Returns: List of matching road segments with current state

### Bike Sharing Tools

- **find_available_bikes**
  - Find bike stations with available bikes
  - Inputs: 
    - `min_bikes` (integer, default: 1) - Minimum number required
    - `near_address` (string, optional) - Address to search near
  - Returns: List of stations matching criteria with availability info

- **get_bike_station_status**
  - Get detailed status of bike stations
  - Input: `station_id` (string, optional) - For specific station
  - Returns: Detailed status including bikes available and free slots

### Air Quality Tools

- **get_air_quality_summary**
  - Get overview of air quality across Valencia
  - Returns: Summary statistics of pollutant levels and air quality ratings

- **find_nearest_air_station**
  - Find closest air quality monitoring station to coordinates
  - Inputs:
    - `latitude` (float) - Latitude coordinate
    - `longitude` (float) - Longitude coordinate
  - Returns: Nearest station details with distance and current readings

- **get_station_data**
  - Get detailed information about air quality monitoring stations
  - Input: `station_name` (string) - Name to search for (partial match)
  - Returns: Detailed pollutant data for matching stations

- **get_pollutant_levels**
  - Get levels of a specific pollutant across all stations
  - Input: `pollutant` (string) - Pollutant type (so2, no2, o3, co, pm10, pm25)
  - Returns: Measurements of specified pollutant with statistics

- **get_air_quality_map**
  - Get geospatial data of air quality stations for mapping
  - Returns: Location data with quality ratings and pollutant levels

### Weather Tools

- **get_current_weather**
  - Get current weather conditions in Valencia
  - Returns: Current weather data including temperature, humidity, wind speed, and weather description

- **get_weather_forecast**
  - Get weather forecast for Valencia
  - Input: `days` (integer, default: 3) - Number of days to forecast (1-7)
  - Returns: Hourly weather forecasts for the requested period

- **get_historical_weather**
  - Get historical weather data for Valencia
  - Inputs:
    - `start_date` (string) - Start date in YYYY-MM-DD format
    - `end_date` (string) - End date in YYYY-MM-DD format
  - Returns: Historical weather data for the specified period

- **get_daily_weather_summary**
  - Get daily weather summary for Valencia
  - Input: `day_offset` (integer, default: 0) - Day offset (0=today, 1=tomorrow, etc.)
  - Returns: Daily weather summary with min/max/avg values

## Claude Desktop Integration

Add this to your `claude_desktop_config.json` (typically found in `%APPDATA%\Claude` on Windows or `~/Library/Application Support/Claude` on macOS):

```json
{
  "mcpServers": {
    "valenciaSmartCity": {
      "command": "uv",
      "args": ["run", "valencia_traffic_mcp.py"]
    }
  }
}
```

Alternatively, install directly with:

```bash
mcp install valencia_traffic_mcp.py --name "Valencia Smart City"
```

## Development

Run the server directly:

```bash
# Run with MCP Inspector for development
mcp dev valencia_traffic_mcp.py

# Run directly (stdio transport)
python valencia_traffic_mcp.py
```

## Example Queries

- "How's the traffic in Valencia right now?"
- "Are there any congested streets in downtown Valencia?"
- "Where can I find a Valenbisi bike near the City Hall?"
- "What's the air quality like in Valencia today?"
- "Which areas of Valencia have the highest pollution levels?"
- "What's the nearest air quality monitoring station to the train station?"
- "Show me the NO2 levels across Valencia"
- "What's the weather forecast for Valencia this week?"
- "Will it rain tomorrow in Valencia?"
- "What were the temperatures in Valencia last week?"
- "Is it a good day for cycling in Valencia today?"

## Data Sources

This server connects to the following data sources:

- [Valencia's open data platform](https://valencia.opendatasoft.com):
  - Traffic data: `estat-transit-temps-real-estado-trafico-tiempo-real`
  - Bike stations: `valenbisi-disponibilitat-valenbisi-disponibilidad`
  - Air quality: `estacions-contaminacio-atmosferiques-estaciones-contaminacion-atmosfericas`

- [Open-Meteo API](https://open-meteo.com/):
  - Current weather data
  - Weather forecast
  - Historical weather data

## License

MIT
