#!/usr/bin/env python3
"""
MCP Server with Country Info and Weather Tools

This server provides two tools:
1. get_country_info: Retrieves country information from restcountries.com
2. get_weather: Retrieves weather information from wttr.in

Usage:
    python mcp_server.py
"""

import json
import sys
import asyncio
import httpx
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent


# Initialize the MCP server
app = Server("country-weather-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_country_info",
            description="Get information about a country including its capital, population, region, and more. Provide the country name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "country_name": {
                        "type": "string",
                        "description": "The name of the country to get information about"
                    }
                },
                "required": ["country_name"]
            }
        ),
        Tool(
            name="get_weather",
            description="Get current weather information for a location. Provide a city name or location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or location to get weather information for"
                    }
                },
                "required": ["location"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "get_country_info":
        country_name = arguments.get("country_name")
        if not country_name:
            return [TextContent(
                type="text",
                text="Error: country_name is required"
            )]
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://restcountries.com/v3.1/name/{country_name}"
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                
                data = response.json()
                if not data:
                    return [TextContent(
                        type="text",
                        text=f"No information found for country: {country_name}"
                    )]
                
                # Get the first result
                country = data[0]
                
                # Extract relevant information
                info = {
                    "name": country.get("name", {}).get("common", "N/A"),
                    "official_name": country.get("name", {}).get("official", "N/A"),
                    "capital": country.get("capital", ["N/A"])[0] if country.get("capital") else "N/A",
                    "population": country.get("population", "N/A"),
                    "region": country.get("region", "N/A"),
                    "subregion": country.get("subregion", "N/A"),
                    "area": country.get("area", "N/A"),
                    "languages": ", ".join(country.get("languages", {}).values()) if country.get("languages") else "N/A",
                    "currencies": ", ".join([f"{v.get('name')} ({k})" for k, v in country.get("currencies", {}).items()]) if country.get("currencies") else "N/A",
                    "timezones": ", ".join(country.get("timezones", [])) if country.get("timezones") else "N/A"
                }
                
                result = json.dumps(info, indent=2)
                return [TextContent(type="text", text=result)]
                
        except httpx.HTTPError as e:
            return [TextContent(
                type="text",
                text=f"Error fetching country information: {str(e)}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Unexpected error: {str(e)}"
            )]
    
    elif name == "get_weather":
        location = arguments.get("location")
        if not location:
            return [TextContent(
                type="text",
                text="Error: location is required"
            )]
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://wttr.in/{location}?format=j1"
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract current weather information
                current = data.get("current_condition", [{}])[0]
                nearest_area = data.get("nearest_area", [{}])[0]
                
                weather_info = {
                    "location": nearest_area.get("areaName", [{}])[0].get("value", location),
                    "country": nearest_area.get("country", [{}])[0].get("value", "N/A"),
                    "temperature_c": current.get("temp_C", "N/A"),
                    "temperature_f": current.get("temp_F", "N/A"),
                    "feels_like_c": current.get("FeelsLikeC", "N/A"),
                    "feels_like_f": current.get("FeelsLikeF", "N/A"),
                    "weather_desc": current.get("weatherDesc", [{}])[0].get("value", "N/A"),
                    "humidity": current.get("humidity", "N/A"),
                    "wind_speed_kmph": current.get("windspeedKmph", "N/A"),
                    "wind_direction": current.get("winddir16Point", "N/A"),
                    "pressure_mb": current.get("pressure", "N/A"),
                    "visibility_km": current.get("visibility", "N/A"),
                    "uv_index": current.get("uvIndex", "N/A")
                }
                
                result = json.dumps(weather_info, indent=2)
                return [TextContent(type="text", text=result)]
                
        except httpx.HTTPError as e:
            return [TextContent(
                type="text",
                text=f"Error fetching weather information: {str(e)}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Unexpected error: {str(e)}"
            )]
    
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
