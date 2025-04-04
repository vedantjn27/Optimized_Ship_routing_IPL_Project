import streamlit as st
import folium
from streamlit_folium import st_folium  # Changed from  folium_static to st_folium
import numpy as np
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import json
import base64
import tempfile
import webbrowser
import os

# Set page configuration
st.set_page_config(
    page_title="Maritime Route Optimizer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a more professional look
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #1E64A8;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #333333; /* Changed to darker color for better visibility */
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0066CC;
    }
    .stButton>button {
        background-color: #1E64A8;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0D4D8C;
    }
    .card h3, .card h4, .card h5 {
        color: #1E3A8A; /* Darker color for headings inside cards */
    }
    .card p, .card li {
        color: #333333; /* Darker color for text inside cards */
    }
</style>
""", unsafe_allow_html=True)

# Classes from original code
class RouteObjective(Enum):
    TIME = auto()
    COST = auto()
    SAFETY = auto()

@dataclass
class Waypoint:
    name: str
    latitude: float
    longitude: float
    weather_risk: float = 0.0
    piracy_risk: float = 0.0
    maritime_traffic: float = 0.0

@dataclass
class ShipParameters:
    ship_type: str
    fuel_efficiency: float
    max_speed: float
    cargo_capacity: float

class MaritimeRouteOptimizer:
    def __init__(self, 
                 population_size: int = 100, 
                 max_generations: int = 50,
                 mutation_rate: float = 0.1):
        self.population_size = population_size
        self.max_generations = max_generations
        self.base_mutation_rate = mutation_rate
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.objectives = {
            RouteObjective.TIME.value: 0.4,
            RouteObjective.COST.value: 0.3,
            RouteObjective.SAFETY.value: 0.3
        }

        self.ports = [
            Waypoint("Chennai", 13.0827, 80.2707),
            Waypoint("Mumbai", 19.0760, 72.8777),
            Waypoint("Colombo", 6.9271, 79.8612),
            Waypoint("Kochi", 9.9312, 76.2673),
            Waypoint("Male", 4.1755, 73.5093),
            Waypoint("Durban", -29.8587, 31.0218),
            Waypoint("Muscat", 23.5880, 58.3829),
            Waypoint("Mombasa", -4.0435, 39.6682),
            Waypoint("Port Louis", -20.1619, 57.4989),
            Waypoint("Jakarta", -6.2088, 106.8456)
        ]
    
    def generate_initial_population(self, 
                                    start: Waypoint, 
                                    end: Waypoint, 
                                    num_waypoints: int = 10) -> List[List[Waypoint]]:
        population = []
        for _ in range(self.population_size):
            route = [start]
            for _ in range(num_waypoints - 2):
                waypoint = random.choice(self.ports)
                if waypoint not in route:
                    waypoint.weather_risk = random.uniform(0, 1)
                    waypoint.piracy_risk = random.uniform(0, 1)
                    waypoint.maritime_traffic = random.uniform(0, 1)
                    route.append(waypoint)
            route.append(end)
            population.append(route)
        return population
    
    def calculate_route_fitness(self, route: List[Waypoint], ship: ShipParameters) -> Dict[int, float]:
        def haversine_distance(wp1: Waypoint, wp2: Waypoint) -> float:
            R = 6371
            lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
            lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        total_distance = sum(haversine_distance(route[i], route[i+1]) for i in range(len(route)-1))
        time_cost = total_distance / ship.max_speed
        fuel_cost = total_distance * (1 / ship.fuel_efficiency)
        maintenance_cost = total_distance * 0.1
        safety_risks = [
            wp.weather_risk * 0.4 + 
            wp.piracy_risk * 0.4 + 
            wp.maritime_traffic * 0.2 
            for wp in route
        ]
        safety_cost = np.mean(safety_risks)
        return {
            RouteObjective.TIME.value: time_cost,
            RouteObjective.COST.value: fuel_cost + maintenance_cost,
            RouteObjective.SAFETY.value: safety_cost
        }

    def optimize_route(self, 
                       start: Waypoint, 
                       end: Waypoint, 
                       ship: ShipParameters,
                       progress_bar=None) -> List[Waypoint]:
        population = self.generate_initial_population(start, end)
        generation_data = []
        
        for generation in range(self.max_generations):
            fitness_scores = [
                self.calculate_route_fitness(route, ship) 
                for route in population
            ]
            
            # Store generation data for visualization
            best_fitness = min([sum(self.objectives[obj] * score for obj, score in scores.items()) for scores in fitness_scores])
            avg_fitness = np.mean([sum(self.objectives[obj] * score for obj, score in scores.items()) for scores in fitness_scores])
            generation_data.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            ranked_population = sorted(
                zip(population, fitness_scores), 
                key=lambda x: sum(
                    self.objectives[obj] * score 
                    for obj, score in x[1].items()
                )
            )
            selected_population = [route for route, _ in ranked_population[:self.population_size//2]]
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
                new_population.append(child)
            population = new_population
            if progress_bar:
                progress_bar.progress((generation + 1) / self.max_generations)
        
        best_route = min(
            population, 
            key=lambda route: sum(
                self.objectives[obj] * score 
                for obj, score in self.calculate_route_fitness(route, ship).items()
            )
        )
        return best_route, generation_data

def haversine_distance(wp1, wp2):
    R = 6371
    lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
    lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
def display_route_on_map(route: List[Waypoint], weather_data=None, piracy_data=None):
    """Simplified map display function with error handling"""
    try:
        # Center map on the approximate middle of the route
        avg_lat = sum(wp.latitude for wp in route) / len(route)
        avg_lon = sum(wp.longitude for wp in route) / len(route)
        
        # Create a basic map with simpler configuration
        map_route = folium.Map(
            location=[avg_lat, avg_lon], 
            zoom_start=4,
            tiles="CartoDB positron",
            control_scale=True
        )

        # Add simplified route line
        points = [[wp.latitude, wp.longitude] for wp in route]
        folium.PolyLine(
            locations=points,
            color="#1E64A8",
            weight=4,
            opacity=0.8
        ).add_to(map_route)

        # Add simplified markers
        for i, wp in enumerate(route):
            if i == 0:  # Start point
                icon_color = "green"
                icon_name = "play"
            elif i == len(route) - 1:  # End point
                icon_color = "red"
                icon_name = "flag"
            else:  # Intermediate points
                icon_color = "blue"
                icon_name = "ship"
            
            folium.Marker(
                location=[wp.latitude, wp.longitude],
                popup=f"<b>{wp.name}</b><br>"
                      f"Weather Risk: {wp.weather_risk:.2f}<br>"
                      f"Piracy Risk: {wp.piracy_risk:.2f}<br>"
                      f"Traffic: {wp.maritime_traffic:.2f}",
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
            ).add_to(map_route)

        return map_route

    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        # Return a basic map as fallback
        return folium.Map(location=[20, 80], zoom_start=3)


def display_simulation_progress(gen_data):
    df = pd.DataFrame(gen_data)
    
    fig = px.line(df, x="generation", y=["best_fitness", "avg_fitness"], 
                 title="Optimization Progress by Generation",
                 labels={"value": "Fitness Score (lower is better)", "generation": "Generation", "variable": "Metric"},
                 color_discrete_map={"best_fitness": "#0066CC", "avg_fitness": "#FF6B6B"})
    
    fig.update_layout(
        legend_title="Metrics",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        font=dict(size=12),
        height=400
    )
    
    return fig

def generate_risk_analysis(route: List[Waypoint]) -> pd.DataFrame:
    risk_data = []
    for i, wp in enumerate(route):
        risk_data.append({
            "Waypoint": wp.name,
            "Weather Risk": wp.weather_risk,
            "Piracy Risk": wp.piracy_risk,
            "Maritime Traffic": wp.maritime_traffic,
            "Total Risk": wp.weather_risk * 0.4 + wp.piracy_risk * 0.4 + wp.maritime_traffic * 0.2,
            "Order": i
        })
    return pd.DataFrame(risk_data)

def simulate_alternative_routes(optimizer, start, end, ship_params, num_simulations=5):
    routes = []
    for _ in range(num_simulations):
        # Adjust weights for different optimization priorities
        optimizer.objectives = {
            RouteObjective.TIME.value: random.uniform(0.2, 0.6),
            RouteObjective.COST.value: random.uniform(0.2, 0.5),
            RouteObjective.SAFETY.value: random.uniform(0.2, 0.5)
        }
        # Normalize weights
        total = sum(optimizer.objectives.values())
        for key in optimizer.objectives:
            optimizer.objectives[key] /= total
            
        route, _ = optimizer.optimize_route(start, end, ship_params)
        
        # Calculate metrics
        fitness = optimizer.calculate_route_fitness(route, ship_params)
        total_distance = sum(haversine_distance(route[i], route[i+1]) for i in range(len(route)-1))
        
        routes.append({
            "route": route,
            "time_priority": optimizer.objectives[RouteObjective.TIME.value],
            "cost_priority": optimizer.objectives[RouteObjective.COST.value],
            "safety_priority": optimizer.objectives[RouteObjective.SAFETY.value],
            "distance": total_distance,
            "time": fitness[RouteObjective.TIME.value],
            "cost": fitness[RouteObjective.COST.value],
            "safety_risk": fitness[RouteObjective.SAFETY.value],
            "waypoints": len(route)
        })
    
    return routes

def display_routes_comparison(routes):
    df = pd.DataFrame([{
        "Route": f"Route {i+1}",
        "Time Priority": r["time_priority"],
        "Cost Priority": r["cost_priority"],
        "Safety Priority": r["safety_priority"],
        "Distance (km)": r["distance"],
        "Est. Time (h)": r["time"],
        "Est. Cost ($)": r["cost"],
        "Safety Risk": r["safety_risk"],
    } for i, r in enumerate(routes)])
    
    # Radar chart for priorities
    fig = go.Figure()
    
    for i, route in enumerate(df["Route"]):
        fig.add_trace(go.Scatterpolar(
            r=[df.loc[i, "Time Priority"], df.loc[i, "Cost Priority"], df.loc[i, "Safety Priority"]],
            theta=["Time", "Cost", "Safety"],
            fill='toself',
            name=route
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 0.7]
            )),
        showlegend=True,
        title="Route Optimization Priorities"
    )
    
    # Bar chart for metrics
    metrics_fig = px.bar(
        df.melt(id_vars=["Route"], value_vars=["Distance (km)", "Est. Time (h)", "Est. Cost ($)", "Safety Risk"],
               var_name="Metric", value_name="Value"),
        x="Route", y="Value", color="Metric", barmode="group",
        title="Route Metrics Comparison"
    )
    
    metrics_fig.update_layout(height=400)
    
    return fig, metrics_fig, df

def create_weather_forecast(route):
    # Simulated weather data for the route points
    weather_data = []
    for wp in route:
        # Create several data points around each waypoint to simulate weather patterns
        for _ in range(5):
            # Random offset to create a pattern around the waypoint
            lat_offset = random.uniform(-2, 2)
            lon_offset = random.uniform(-2, 2)
            # Weather intensity based on the waypoint's weather risk but with some randomness
            intensity = wp.weather_risk * random.uniform(0.8, 1.2)
            weather_data.append([wp.latitude + lat_offset, wp.longitude + lon_offset, intensity])
    
    return weather_data

def create_piracy_hotspots(route):
    # Simulated piracy risk zones
    piracy_data = []
    for wp in route:
        if wp.piracy_risk > 0.3:  # Only create hotspots for higher risk areas
            piracy_data.append({
                'location': [wp.latitude, wp.longitude],
                'intensity': wp.piracy_risk
            })
    
    return piracy_data

def format_time(hours):
    """
    Format hours into days and hours (e.g., "2d 5h")
    Handles both float and int types for hours.
    """
    if hours is None:
        return "N/A"
    
    try:
        hours_float = float(hours)
        days = int(hours_float // 24)
        remaining_hours = int(hours_float % 24)
        if days > 0:
            return f"{days}d {remaining_hours}h"
        else:
            return f"{remaining_hours}h"
    except (TypeError, ValueError):
        return "Error"
    
def main():
    st.markdown('<h1 class="main-header">🚢 Maritime Route Optimizer</h1>', unsafe_allow_html=True)
    
    # Initialize optimizer
    optimizer = MaritimeRouteOptimizer()
    
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Port Selection
        st.subheader("Route Selection")
        port_names = [port.name for port in optimizer.ports]
        start_port = st.selectbox("Start Port", port_names, index=0)
        end_port = st.selectbox("End Port", port_names, index=1)
        
        # Ship Selection
        st.subheader("Ship Parameters")
        ship_types = ["Cargo", "Container", "Tanker", "Passenger"]
        ship_type = st.selectbox("Ship Type", ship_types, index=0)
        
        # Advanced Settings
        st.subheader("Advanced Settings")
        with st.expander("Algorithm Parameters"):
            population = st.slider("Population Size", min_value=50, max_value=200, value=100, step=10)
            generations = st.slider("Max Generations", min_value=10, max_value=100, value=50, step=5)
            mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        with st.expander("Optimization Priorities"):
            time_priority = st.slider("Time Priority", min_value=0.1, max_value=1.0, value=0.4, step=0.1)
            cost_priority = st.slider("Cost Priority", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
            safety_priority = st.slider("Safety Priority", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
            
            # Normalize weights to sum to 1
            total = time_priority + cost_priority + safety_priority
            time_priority /= total
            cost_priority /= total 
            safety_priority /= total
            
            st.caption(f"Normalized weights: Time={time_priority:.2f}, Cost={cost_priority:.2f}, Safety={safety_priority:.2f}")
        
        # Set optimizer parameters
        optimizer.population_size = population
        optimizer.max_generations = generations
        optimizer.base_mutation_rate = mutation_rate
        optimizer.objectives = {
            RouteObjective.TIME.value: time_priority,
            RouteObjective.COST.value: cost_priority,
            RouteObjective.SAFETY.value: safety_priority
        }
        
        optimize_button = st.button("Optimize Route", use_container_width=True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Route Map", "Route Analysis", "Comparison", "Simulation"])
    
    if optimize_button:
        with st.spinner("Optimizing route..."):
            # Find waypoint objects
            start_waypoint = next(port for port in optimizer.ports if port.name == start_port)
            end_waypoint = next(port for port in optimizer.ports if port.name == end_port)
            
            # Ship parameters based on type
            ship_params = {
                "Cargo": ShipParameters(ship_type="Cargo", fuel_efficiency=0.8, max_speed=20.0, cargo_capacity=5000.0),
                "Container": ShipParameters(ship_type="Container", fuel_efficiency=0.7, max_speed=18.0, cargo_capacity=8000.0),
                "Tanker": ShipParameters(ship_type="Tanker", fuel_efficiency=0.6, max_speed=15.0, cargo_capacity=10000.0),
                "Passenger": ShipParameters(ship_type="Passenger", fuel_efficiency=0.5, max_speed=22.0, cargo_capacity=2000.0)
            }
            
            # Show progress bar
            progress_text = "Optimizing route with genetic algorithm..."
            progress_bar = st.progress(0, text=progress_text)
            
            # Optimize route
            optimized_route, generation_data = optimizer.optimize_route(
                start_waypoint, end_waypoint, ship_params[ship_type], progress_bar)
            
            # Calculate route fitness
            fitness = optimizer.calculate_route_fitness(optimized_route, ship_params[ship_type])
            
            # Calculate total distance
            total_distance = sum(haversine_distance(optimized_route[i], optimized_route[i+1]) 
                              for i in range(len(optimized_route)-1))
            
            # Generate simulation data for other tabs
            weather_data = create_weather_forecast(optimized_route)
            piracy_data = create_piracy_hotspots(optimized_route)
            
            # Generate alternative routes for comparison
            alternative_routes = simulate_alternative_routes(
                optimizer, start_waypoint, end_waypoint, ship_params[ship_type])
            
            # Save results in session state
            st.session_state.optimized_route = optimized_route
            st.session_state.fitness = fitness
            st.session_state.total_distance = total_distance
            st.session_state.ship_params = ship_params[ship_type]
            st.session_state.generation_data = generation_data
            st.session_state.weather_data = weather_data
            st.session_state.piracy_data = piracy_data
            st.session_state.alternative_routes = alternative_routes
            st.session_state.ship_type = ship_type
            
            # Initialize simulation results if not exists
            if 'simulation_results' not in st.session_state:
                st.session_state.simulation_results = None
            
            progress_bar.empty()
            st.success("Route optimization completed!")

    # Display results if available
    if 'optimized_route' in st.session_state:
        optimized_route = st.session_state.optimized_route
        fitness = st.session_state.fitness
        total_distance = st.session_state.total_distance
        ship_params = st.session_state.ship_params
        generation_data = st.session_state.generation_data
        weather_data = st.session_state.weather_data
        piracy_data = st.session_state.piracy_data
        alternative_routes = st.session_state.alternative_routes
        
        with tab1:
            st.markdown('<h2 class="sub-header">Optimized Maritime Route</h2>', unsafe_allow_html=True)
    
            # Key metrics in colorful cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="card" style="background-color: #e6f2ff;">
                    <h3 style="margin:0;">Distance</h3>
                    <p class="metric-value">{total_distance:.1f} km</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                time_value = fitness.get(RouteObjective.TIME.value, 0)
                st.markdown(f"""
                <div class="card" style="background-color: #e6fffa;">
                    <h3 style="margin:0;">Est. Time</h3>
                    <p class="metric-value">{format_time(time_value)}</p>
                </div>
                 """, unsafe_allow_html=True)
            with col3:
                cost_value = fitness.get(RouteObjective.COST.value, 0)
                st.markdown(f"""
                <div class="card" style="background-color: #fff5e6;">
                    <h3 style="margin:0;">Est. Cost</h3>
                    <p class="metric-value">${cost_value:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                safety_value = fitness.get(RouteObjective.SAFETY.value, 0)
                safety_percentage = 100 - (safety_value * 100)
                st.markdown(f"""
                <div class="card" style="background-color: #f0f5e6;">
                    <h3 style="margin:0;">Safety Index</h3>
                    <p class="metric-value">{safety_percentage:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            # Map options
            map_options = st.multiselect(
                "Map Overlays",
                ["Weather Risk Heatmap", "Piracy Risk Zones"],
                default=["Piracy Risk Zones"]
            )
            
            try:
            # Generate the map
                route_map = display_route_on_map(optimized_route)
        
                # Display with st_folium and explicit size
                st_folium(
                    route_map,
                    width=1000,
                    height=600,
                    returned_objects=[],
                    key=f"main_route_{start_port}_{end_port}"  # Unique based on ports
                )
        
            except Exception as e:
                st.error(f"Error displaying map: {str(e)}")
                st.warning("Showing fallback map instead")
                fallback_map = folium.Map(location=[20, 80], zoom_start=3)
                st_folium(
                    fallback_map,
                    width=1000,
                    height=600,
                    key="fallback_map"
                )

        with tab2:
            st.markdown('<h2 class="sub-header">Route Analysis Dashboard</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Optimization progress chart
                st.plotly_chart(display_simulation_progress(generation_data), use_container_width=True)
                
                # Risk analysis along the route
                risk_df = generate_risk_analysis(optimized_route)
                
                risk_chart = px.line(risk_df, x="Order", y=["Weather Risk", "Piracy Risk", "Maritime Traffic", "Total Risk"],
                             markers=True, line_shape="linear", title="Risk Analysis Along Route",
                             labels={"Order": "Waypoint Order", "value": "Risk Level"})
                
                risk_chart.update_layout(xaxis = dict(
                    tickmode = 'array',
                    tickvals = risk_df["Order"],
                    ticktext = risk_df["Waypoint"]
                ))
                
                st.plotly_chart(risk_chart, use_container_width=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Route Details")
                
                # Ship details
                st.markdown(f"**Ship Type:** {st.session_state.ship_type}")
                st.markdown(f"**Max Speed:** {ship_params.max_speed} knots")
                st.markdown(f"**Fuel Efficiency:** {ship_params.fuel_efficiency}")
                st.markdown(f"**Cargo Capacity:** {ship_params.cargo_capacity} tons")
                
                st.markdown("**Port Sequence:**")
                for i, wp in enumerate(optimized_route):
                    emoji = "🚢" if i == 0 else "🏁" if i == len(optimized_route) - 1 else "📍"
                    st.markdown(f"{emoji} {i+1}. **{wp.name}**")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Weather and piracy breakdown
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Risk Assessment")
                
                # Average risks
                avg_weather = np.mean([wp.weather_risk for wp in optimized_route])
                avg_piracy = np.mean([wp.piracy_risk for wp in optimized_route])
                avg_traffic = np.mean([wp.maritime_traffic for wp in optimized_route])
                
                # Risk gauge charts
                risk_gauges = go.Figure()
                
                risk_gauges.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = avg_weather * 100,
                    title = {'text': "Weather Risk"},
                    gauge = {'axis': {'range': [0, 100]},
                             'bar': {'color': "blue"},
                             'steps': [
                                 {'range': [0, 33], 'color': "lightgreen"},
                                 {'range': [33, 66], 'color': "yellow"},
                                 {'range': [66, 100], 'color': "salmon"}],
                             'threshold': {
                                 'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75,
                                 'value': avg_weather * 100}}))
                
                st.plotly_chart(risk_gauges, use_container_width=True)
                
                # Highest risk areas
                max_weather_wp = max(optimized_route, key=lambda wp: wp.weather_risk)
                max_piracy_wp = max(optimized_route, key=lambda wp: wp.piracy_risk)
                
                st.markdown(f"⚠️ **Highest Weather Risk:** {max_weather_wp.name} ({max_weather_wp.weather_risk:.2f})")
                st.markdown(f"⚠️ **Highest Piracy Risk:** {max_piracy_wp.name} ({max_piracy_wp.piracy_risk:.2f})")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ETA Calculator
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("ETA Calculator")
                
                departure_date = st.date_input("Departure Date", datetime.now())
                departure_time = st.time_input("Departure Time", datetime.now().time())
                
                # Calculate ETA
                departure_datetime = datetime.combine(departure_date, departure_time)
                travel_hours = fitness[RouteObjective.TIME.value]
                eta = departure_datetime + timedelta(hours=travel_hours)
                
                st.markdown(f"**Estimated Time of Arrival:** {eta.strftime('%Y-%m-%d %H:%M')}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h2 class="sub-header">Route Comparison</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <p>This tab compares your optimized route with alternative routes using different optimization priorities.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display radar chart and bar chart for route comparison
            radar_chart, bar_chart, comparison_df = display_routes_comparison(alternative_routes)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(radar_chart, use_container_width=True)
            with col2:
                st.plotly_chart(bar_chart, use_container_width=True)
            
            # Show detailed comparison table
            st.markdown('<h3 class="sub-header">Detailed Comparison</h3>', unsafe_allow_html=True)
            st.dataframe(comparison_df.style.highlight_min(subset=['Distance (km)', 'Est. Time (h)', 'Est. Cost ($)', 'Safety Risk'], color='lightgreen')
                         .highlight_max(subset=['Distance (km)', 'Est. Time (h)', 'Est. Cost ($)', 'Safety Risk'], color='salmon')
                         .format({
                             'Distance (km)': '{:.1f}',
                             'Est. Time (h)': '{:.1f}',
                             'Est. Cost ($)': '${:.2f}',
                             'Safety Risk': '{:.3f}',
                             'Time Priority': '{:.2f}',
                             'Cost Priority': '{:.2f}',
                             'Safety Priority': '{:.2f}'
                         }))
            
            # Route visualization selector
            st.markdown('<h3 class="sub-header">Visual Comparison</h3>', unsafe_allow_html=True)
            selected_route = st.selectbox(
                "Select Alternative Route to Visualize",
                [f"Route {i+1}" for i in range(len(alternative_routes))]
            )
            
            selected_idx = int(selected_route.split()[1]) - 1
            selected_route_data = alternative_routes[selected_idx]
            
            # Show selected alternative route on map
            alt_map = display_route_on_map(selected_route_data["route"])
            st_folium(
                alt_map,
                width=1000,
                height=400,
                key=f"alt_route_{selected_idx}"  # Unique based on selection
            )
        
        with tab4:
            st.markdown('<h2 class="sub-header">Interactive Simulation</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <p>This simulation allows you to explore how different factors affect your maritime route.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sim_col1, sim_col2 = st.columns([1, 2])
            
            with sim_col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Simulation Parameters")
                
                # Scenario selection
                scenario = st.selectbox(
                    "Select Scenario",
                    ["Normal Conditions", "High Piracy Risk", "Extreme Weather", "Fuel Shortage", "Custom"]
                )
                
                # Set default values based on scenario
                if scenario == "Normal Conditions":
                    weather_factor = 0.3
                    piracy_factor = 0.3
                    fuel_cost = 1.0
                elif scenario == "High Piracy Risk":
                    weather_factor = 0.3
                    piracy_factor = 0.8
                    fuel_cost = 1.0
                elif scenario == "Extreme Weather":
                    weather_factor = 0.8
                    piracy_factor = 0.3
                    fuel_cost = 1.0
                elif scenario == "Fuel Shortage":
                    weather_factor = 0.3
                    piracy_factor = 0.3
                    fuel_cost = 2.5
                else:  # Custom
                    weather_factor = st.slider("Weather Severity", 0.1, 1.0, 0.3, 0.1)
                    piracy_factor = st.slider("Piracy Risk Level", 0.1, 1.0, 0.3, 0.1)
                    fuel_cost = st.slider("Fuel Cost Multiplier", 0.5, 3.0, 1.0, 0.1)
                
                # Display the values
                st.markdown(f"**Weather Severity:** {weather_factor:.1f}")
                st.markdown(f"**Piracy Risk Level:** {piracy_factor:.1f}")
                st.markdown(f"**Fuel Cost Multiplier:** {fuel_cost:.1f}x")
                
                # Simulation controls
                st.markdown("### Simulation Controls")
                sim_button = st.button("Run Simulation", use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with sim_col2:
                # Check if simulation results exist in session state
                if st.session_state.simulation_results is not None and not sim_button:
                    # Display cached results
                    comparison_data = st.session_state.simulation_results['comparison_data']
                    updated_weather = st.session_state.simulation_results['updated_weather']
                    updated_piracy = st.session_state.simulation_results['updated_piracy']
                    scenario = st.session_state.simulation_results['scenario']
                    
                    # Display comparison chart
                    fig = px.bar(comparison_data, x="Metric", y=["Original", "Simulated"], barmode="group",
                                title=f"Impact of {scenario} on Route Metrics",
                                labels={"value": "Value", "variable": "Condition"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display percentage changes
                    st.markdown('<h3 class="sub-header">Impact Analysis</h3>', unsafe_allow_html=True)
                    
                    for idx, row in comparison_data.iterrows():
                        change = row["Change (%)"]
                        color = "red" if change > 0 else "green"
                        direction = "increase" if change > 0 else "decrease"
                        st.markdown(f"**{row['Metric']}**: {abs(change):.1f}% {direction} <span style='color:{color};'>{'↑' if change > 0 else '↓'}</span>", 
                                   unsafe_allow_html=True)
                    
                    # Risk heat map with simulated conditions
                    st.markdown('<h3 class="sub-header">Updated Risk Map</h3>', unsafe_allow_html=True)
                    
                    # Show updated map
                    sim_map = display_route_on_map(optimized_route, updated_weather, updated_piracy)
                    st_folium(
                        sim_map,
                        width=800,
                        height=400,
                        key=f"sim_map_{scenario}"  # Unique based on scenario
                    )
                
                if sim_button:
                    with st.spinner("Running simulation..."):
                        # Create a copy of the route to avoid modifying the original
                        simulated_route = [Waypoint(wp.name, wp.latitude, wp.longitude, 
                                                  wp.weather_risk, wp.piracy_risk, wp.maritime_traffic) 
                                         for wp in optimized_route]
                        
                        # Adjust route parameters based on scenario
                        for wp in simulated_route:
                            wp.weather_risk = min(1.0, wp.weather_risk * weather_factor * 1.5)
                            wp.piracy_risk = min(1.0, wp.piracy_risk * piracy_factor * 1.5)
                        
                        # Recalculate with adjusted parameters
                        adjusted_fitness = optimizer.calculate_route_fitness(simulated_route, ship_params)
                        adjusted_fitness[RouteObjective.COST.value] *= fuel_cost
                        
                        # Original vs. Simulated comparison
                        comparison_data = pd.DataFrame({
                            "Metric": ["Time (hours)", "Cost ($)", "Safety Risk"],
                            "Original": [fitness[RouteObjective.TIME.value], 
                                        fitness[RouteObjective.COST.value], 
                                        fitness[RouteObjective.SAFETY.value]],
                            "Simulated": [adjusted_fitness[RouteObjective.TIME.value], 
                                         adjusted_fitness[RouteObjective.COST.value], 
                                         adjusted_fitness[RouteObjective.SAFETY.value]]
                        })
                        
                        # Calculate percentage changes
                        comparison_data["Change (%)"] = ((comparison_data["Simulated"] - comparison_data["Original"]) / 
                                                       comparison_data["Original"] * 100)
                        
                        # Generate updated weather and piracy data
                        updated_weather = create_weather_forecast(simulated_route)
                        updated_piracy = create_piracy_hotspots(simulated_route)
                        
                        # Store results in session state
                        st.session_state.simulation_results = {
                            'comparison_data': comparison_data,
                            'updated_weather': updated_weather,
                            'updated_piracy': updated_piracy,
                            'scenario': scenario
                        }
                        
                        # Display comparison chart
                        fig = px.bar(comparison_data, x="Metric", y=["Original", "Simulated"], barmode="group",
                                    title=f"Impact of {scenario} on Route Metrics",
                                    labels={"value": "Value", "variable": "Condition"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display percentage changes
                        st.markdown('<h3 class="sub-header">Impact Analysis</h3>', unsafe_allow_html=True)
                        
                        for idx, row in comparison_data.iterrows():
                            change = row["Change (%)"]
                            color = "red" if change > 0 else "green"
                            direction = "increase" if change > 0 else "decrease"
                            st.markdown(f"**{row['Metric']}**: {abs(change):.1f}% {direction} <span style='color:{color};'>{'↑' if change > 0 else '↓'}</span>", 
                                       unsafe_allow_html=True)
                        
                        # Show updated map
                        sim_map = display_route_on_map(simulated_route, updated_weather, updated_piracy)
                        st_folium(
                            sim_map,
                            width=800,
                            height=400,
                            key=f"sim_map_{scenario}"  # Unique based on scenario
                        )
                
                if st.session_state.simulation_results is None and not sim_button:
                    st.info("👈 Configure the simulation parameters and click 'Run Simulation' to see the impact on your route.")
    
    else:
        # Display placeholder content when no route is optimized yet
        st.info("👈 Configure your route parameters in the sidebar and click 'Optimize Route' to begin.")
        
        # Add some placeholder visualizations
        st.markdown('<h2 class="sub-header">Maritime Route Optimization</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h3>How It Works</h3>
                <p>This application uses a genetic algorithm to optimize maritime shipping routes based on multiple objectives:</p>
                <ul>
                    <li><strong>Time:</strong> Minimizing travel time between ports</li>
                    <li><strong>Cost:</strong> Reducing fuel and operational costs</li>
                    <li><strong>Safety:</strong> Avoiding regions with high weather, piracy, or traffic risks</li>
                </ul>
                <p>Configure your route parameters in the sidebar to get started.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Features</h3>
                <ul>
                    <li>Interactive route maps with risk overlays</li>
                    <li>Detailed analytics and optimization metrics</li>
                    <li>Alternative route comparison</li>
                    <li>Scenario simulation with adjustable risk factors</li>
                    <li>ETA calculator with real-time projections</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 50px; padding: 10px; border-top: 1px solid #ccc; text-align: center;">
        <p style="color: #666; font-size: 14px;">Maritime Route Optimizer v1.0 © 2023</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()