# Valencia Smart City MCP Server

This Model Context Protocol (MCP) server provides access to open data from Valencia, currently focusing on real-time traffic information.

## Features

- **Real-time traffic data**: Access current traffic conditions across Valencia
- **Bike sharing availability**: Access real-time Valenbisi bike station status
- **Processed data**: Get human-readable traffic and bike station information
- **Search capabilities**: Find specific road segments or bike stations
- **Congestion analysis**: Get summaries of traffic congestion across the city
- **Bike availability tools**: Find stations with available bikes near specific locations

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

You can run the server directly:

```bash
python valencia_traffic_mcp.py
```

Or use the MCP CLI for development and testing:

```bash
# For development with MCP Inspector
mcp dev valencia_traffic_mcp.py

# For integration with Claude Desktop
mcp install valencia_traffic_mcp.py
```

### Available Resources

- `valencia://traffic/raw` - Raw traffic data from Valencia's API
- `valencia://traffic/processed` - Processed traffic data with human-readable state information
- `valencia://bikes/raw` - Raw bike station data from Valencia's API
- `valencia://bikes/processed` - Processed bike station data with formatted fields

### Available Tools

- `get_traffic_status` - Get current traffic status, optionally filtered by state
- `get_congestion_summary` - Get a summary of current traffic congestion
- `search_road_segment` - Search for specific road segments by name
- `find_available_bikes` - Find bike stations with available bikes, optionally near a specific address
- `get_bike_station_status` - Get detailed status of bike stations, optionally filtered by station ID

### Available Prompts

- `traffic_report` - Generate a comprehensive traffic report
- `navigation_advice` - Provide navigation advice for reaching a specific destination
- `bike_availability` - Check bike availability near a specific location
- `urban_mobility_report` - Create a comprehensive urban mobility report

## Data Sources

This MCP server connects to Valencia's open data platform at:
https://valencia.opendatasoft.com

The data is from the following datasets:

- Traffic data: `estat-transit-temps-real-estado-trafico-tiempo-real`
- Bike stations: `valenbisi-disponibilitat-valenbisi-disponibilidad`

Data is updated every few minutes from official Valencia city sources.

## Traffic State Codes

| Code | Description |
|------|-------------|
| 0 | Fluido |
| 1 | Denso |
| 2 | Congestionado |
| 3 | Cortado |
| 4 | Sin datos |
| 5 | Paso inferior fluido |
| 6 | Paso inferior denso |
| 7 | Paso inferior congestionado |
| 8 | Paso inferior cortado |
| 9 | Sin datos (paso inferior) |

## Future Enhancements

- Add additional data sources from Valencia's open data portal
- Implement historical data analysis
- Add visualization capabilities for traffic patterns
- Expand to other cities with open data portals
