# Valencia Smart City MCP Server

MCP server providing real-time traffic and bike-sharing data from Valencia, Spain for Claude and other LLMs.

## Features

- Real-time traffic conditions across Valencia
- Valenbisi bike station availability
- Traffic congestion analysis
- Search capabilities for specific roads or bike stations

## Requirements

- Python 3.10+
- Claude Desktop or other MCP-compatible client
- Internet connection

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartCityMCP.git
cd SmartCityMCP

# Install dependencies
pip install -r requirements.txt
```

## Tools

The server exposes these tools:

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

## Claude Desktop Integration

Add this to your `claude_desktop_config.json` (typically found in `%APPDATA%\Claude` on Windows or `~/Library/Application Support/Claude` on macOS):

```json
{
  "mcpServers": {
    "valenciaTraffic": {
      "command": "uv",
      "args": ["run", "valencia_traffic_mcp.py"]
    }
  }
}
```

Alternatively, install directly with:

```bash
mcp install valencia_traffic_mcp.py --name "Valencia Traffic"
```

## Development

Run the server directly:

```bash
# Run with MCP Inspector for development
mcp dev valencia_traffic_mcp.py

# Run directly (stdio transport)
python valencia_traffic_mcp.py
```

## Traffic State Codes

| Code | Description |
|------|-------------|
| 0 | Fluido (Flowing) |
| 1 | Denso (Dense) |
| 2 | Congestionado (Congested) |
| 3 | Cortado (Closed) |
| 4 | Sin datos (No data) |

## Example Queries

- "How's the traffic in Valencia right now?"
- "Are there any congested streets in downtown Valencia?"
- "Where can I find a Valenbisi bike near the City Hall?"
- "Show me a summary of the current traffic conditions"

## Data Source

This server connects to [Valencia's open data platform](https://valencia.opendatasoft.com).

## License

MIT