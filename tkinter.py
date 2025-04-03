import tkinter as tk
from tkinter import ttk, messagebox
import folium
import webbrowser
import os
import numpy as np
import random
from typing import List, Dict
from dataclasses import dataclass 
from enum import Enum, auto
import logging
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import io
from ttkthemes import ThemedTk
import json
import sys
import datetime

# Keep all the backend code
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
            RouteObjective.TIME: 0.4,
            RouteObjective.COST: 0.3,
            RouteObjective.SAFETY: 0.3
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
    
    def calculate_route_fitness(self, route: List[Waypoint], ship: ShipParameters) -> Dict[RouteObjective, float]:
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
            RouteObjective.TIME: time_cost,
            RouteObjective.COST: fuel_cost + maintenance_cost,
            RouteObjective.SAFETY: safety_cost
        }

    def optimize_route(self, 
                       start: Waypoint, 
                       end: Waypoint, 
                       ship: ShipParameters,
                       progress_callback=None) -> List[Waypoint]:
        population = self.generate_initial_population(start, end)
        for generation in range(self.max_generations):
            if progress_callback:
                progress_callback(generation / self.max_generations * 100)
                
            fitness_scores = [
                self.calculate_route_fitness(route, ship) 
                for route in population
            ]
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
            self.logger.info(f"Generation {generation + 1} completed")
        
        best_route = min(
            population, 
            key=lambda route: sum(
                self.objectives[obj] * score 
                for obj, score in self.calculate_route_fitness(route, ship).items()
            )
        )
        return best_route

def display_route_on_map(route: List[Waypoint], filename="optimized_route_map.html", show_browser=True):
    map_route = folium.Map(location=[route[0].latitude, route[0].longitude], zoom_start=4)
    
    # Add a better basemap
    folium.TileLayer('cartodbpositron').add_to(map_route)
    
    # Create a feature group for the route
    route_group = folium.FeatureGroup(name="Optimized Route")
    
    # Add start and end points with special markers
    folium.Marker(
        location=[route[0].latitude, route[0].longitude],
        popup=f"<b>START:</b> {route[0].name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(route_group)
    
    folium.Marker(
        location=[route[-1].latitude, route[-1].longitude],
        popup=f"<b>END:</b> {route[-1].name}",
        icon=folium.Icon(color="red", icon="stop", prefix="fa")
    ).add_to(route_group)
    
    # Add waypoints in between
    for i, wp in enumerate(route[1:-1], 1):
        risk_level = wp.weather_risk + wp.piracy_risk
        color = "green" if risk_level < 0.5 else "orange" if risk_level < 1.0 else "red"
        
        folium.Marker(
            location=[wp.latitude, wp.longitude],
            popup=f"""
            <b>{wp.name}</b><br>
            <b>Weather Risk:</b> {wp.weather_risk:.2f}<br>
            <b>Piracy Risk:</b> {wp.piracy_risk:.2f}<br>
            <b>Maritime Traffic:</b> {wp.maritime_traffic:.2f}
            """,
            icon=folium.Icon(color=color, icon="ship", prefix="fa")
        ).add_to(route_group)
    
    # Add the route line
    points = [[wp.latitude, wp.longitude] for wp in route]
    folium.PolyLine(
        locations=points,
        color="#3388ff",
        weight=4,
        opacity=0.8,
        tooltip="Optimized Route"
    ).add_to(route_group)
    
    # Add distance markers
    def haversine_distance(wp1, wp2):
        R = 6371
        lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
        lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    for i in range(len(route)-1):
        midpoint_lat = (route[i].latitude + route[i+1].latitude) / 2
        midpoint_lon = (route[i].longitude + route[i+1].longitude) / 2
        distance = haversine_distance(route[i], route[i+1])
        
        folium.Marker(
            location=[midpoint_lat, midpoint_lon],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(75, 18),
                html=f'<div style="font-size: 10pt; background-color: rgba(255,255,255,0.7); border-radius: 5px; padding: 2px 5px;">{distance:.1f} km</div>'
            )
        ).add_to(route_group)
    
    # Add the feature group to the map
    route_group.add_to(map_route)
    
    # Add layer control
    folium.LayerControl().add_to(map_route)
    
    # Save map
    map_route.save(filename)
    
    if show_browser:
        webbrowser.open('file://' + os.path.realpath(filename))
    
    return map_route

# New modernized UI
class ModernMaritimeRouteOptimizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Maritime Route Optimizer Pro")
        
        # Set theme
        if not isinstance(master, ThemedTk):
            print("Warning: ThemedTk is not being used. Some styling may not work.")
        
        # Configure window
        self.master.geometry("1280x800")
        self.master.minsize(1024, 768)
        
        # Create style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f5f7fa")
        self.style.configure("TLabelframe", background="#f5f7fa")
        self.style.configure("TLabelframe.Label", background="#f5f7fa", foreground="#2d3e50", font=("Segoe UI", 12, "bold"))
        self.style.configure("TLabel", background="#f5f7fa", foreground="#2d3e50", font=("Segoe UI", 10))
        self.style.configure("TButton", background="#3498db", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.style.map("TButton", 
                      background=[("active", "#2980b9"), ("pressed", "#1c6ea4")],
                      foreground=[("active", "#000000"), ("pressed", "#000000")])
        
        # Custom blue button style
        self.style.configure("Blue.TButton", background="#3498db", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.style.map("Blue.TButton", 
                      background=[("active", "#2980b9"), ("pressed", "#1c6ea4")],
                      foreground=[("active", "#000000"), ("pressed", "#000000")])
        
        # Custom green button style
        self.style.configure("Green.TButton", background="#2ecc71", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.style.map("Green.TButton", 
                      background=[("active", "#27ae60"), ("pressed", "#219653")],
                      foreground=[("active", "#000000"), ("pressed", "#000000")])
                      
        # Custom orange button style
        self.style.configure("Orange.TButton", background="#e67e22", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.style.map("Orange.TButton", 
                      background=[("active", "#d35400"), ("pressed", "#a04000")],
                      foreground=[("active", "#000000"), ("pressed", "#000000")])
        
        # Header frame
        self.header_frame = tk.Frame(master, bg="#2d3e50", height=80)
        self.header_frame.pack(fill=tk.X)
        
        # App title
        self.title_label = tk.Label(
            self.header_frame, 
            text="Maritime Route Optimizer Pro",
            font=("Segoe UI", 20, "bold"),
            fg="#ffffff",
            bg="#2d3e50"
        )
        self.title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Current date and time
        self.date_label = tk.Label(
            self.header_frame,
            text=datetime.datetime.now().strftime("%B %d, %Y"),
            font=("Segoe UI", 10),
            fg="#ecf0f1",
            bg="#2d3e50"
        )
        self.date_label.pack(side=tk.RIGHT, padx=20, pady=20)
        
        # Main container with two columns
        self.main_container = tk.Frame(master, bg="#f5f7fa")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column for inputs
        self.left_column = tk.Frame(self.main_container, bg="#f5f7fa", width=350)
        self.left_column.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.left_column.pack_propagate(False)
        
        # Right column for outputs
        self.right_column = tk.Frame(self.main_container, bg="#f5f7fa")
        self.right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize optimizer
        self.optimizer = MaritimeRouteOptimizer()
         # For optimization progress
        self.progress_var = tk.DoubleVar(value=0)
        
        # Create UI components
        self.create_input_section()
        self.create_dashboard_section()
        self.create_map_view()
        
        # Variables to store data
        self.current_route = None
        self.current_fitness = None
        self.current_ship = None
        
        
        # Load ship icons
        self.load_ship_icons()

    def load_ship_icons(self):
        """Load ship icons for the different ship types"""
        self.ship_icons = {}
        
        # Create simple ship icons programmatically since we don't have external images
        ship_colors = {
            "Cargo": "#3498db",  # Blue
            "Container": "#e74c3c",  # Red
            "Tanker": "#2ecc71",  # Green
            "Passenger": "#f39c12"  # Yellow/Orange
        }
        
        for ship_type, color in ship_colors.items():
            # Create a small image using matplotlib
            fig, ax = plt.subplots(figsize=(1, 1), dpi=50)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Simple ship shape
            ship_poly = plt.Polygon([(0.2, 0.3), (0.8, 0.3), (0.9, 0.5), (0.8, 0.7), (0.2, 0.7), (0.1, 0.5)], 
                                    closed=True, color=color)
            ax.add_patch(ship_poly)
            
            # Convert to PhotoImage
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=True)
            buf.seek(0)
            img = Image.open(buf)
            photo_img = ImageTk.PhotoImage(img)
            
            # Store the image reference
            self.ship_icons[ship_type] = photo_img
            
            # Close figure
            plt.close(fig)
    
    def create_input_section(self):
        """Create the input section with route and ship configuration"""
        # Container frame with a nice border and title
        input_frame = ttk.LabelFrame(self.left_column, text="Route Configuration", padding=(10, 5, 10, 10))
        input_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        # Port names extracted from self.optimizer.ports
        ports = [port.name for port in self.optimizer.ports]
        ports.sort()  # Sort alphabetically
        
        # Start Port Selection
        port_frame = ttk.Frame(input_frame)
        port_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(port_frame, text="Start Port:").pack(side=tk.LEFT, padx=5)
        self.start_port_var = tk.StringVar(value=ports[0] if ports else "")
        self.start_port_dropdown = ttk.Combobox(port_frame, textvariable=self.start_port_var, values=ports, state="readonly", width=15)
        self.start_port_dropdown.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # End Port Selection
        port_frame2 = ttk.Frame(input_frame)
        port_frame2.pack(fill=tk.X, pady=5)
        
        ttk.Label(port_frame2, text="End Port:").pack(side=tk.LEFT, padx=5)
        self.end_port_var = tk.StringVar(value=ports[-1] if len(ports) > 1 else "")
        self.end_port_dropdown = ttk.Combobox(port_frame2, textvariable=self.end_port_var, values=ports, state="readonly", width=15)
        self.end_port_dropdown.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Separator
        ttk.Separator(input_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Ship Type Selection
        ship_frame = ttk.LabelFrame(input_frame, text="Ship Configuration", padding=(5, 5, 5, 5))
        ship_frame.pack(fill=tk.X, pady=5)
        
        ship_types = ["Cargo", "Container", "Tanker", "Passenger"]
        self.ship_type_var = tk.StringVar(value="Cargo")
        
        # Create ship selection with radio buttons and icons
        for i, ship_type in enumerate(ship_types):
            ship_btn = ttk.Radiobutton(
                ship_frame, 
                text=ship_type,
                variable=self.ship_type_var,
                value=ship_type,
                command=self.update_ship_details
            )
            ship_btn.grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
        
        # Ship parameters based on type
        self.ship_params = {
            "Cargo": ShipParameters(ship_type="Cargo", fuel_efficiency=0.8, max_speed=20.0, cargo_capacity=5000.0),
            "Container": ShipParameters(ship_type="Container", fuel_efficiency=0.7, max_speed=18.0, cargo_capacity=8000.0),
            "Tanker": ShipParameters(ship_type="Tanker", fuel_efficiency=0.6, max_speed=15.0, cargo_capacity=10000.0),
            "Passenger": ShipParameters(ship_type="Passenger", fuel_efficiency=0.5, max_speed=22.0, cargo_capacity=2000.0)
        }
        
        # Ship details frame
        self.ship_details_frame = ttk.Frame(input_frame)
        self.ship_details_frame.pack(fill=tk.X, pady=10)
        
        # Labels for ship details
        self.speed_label = ttk.Label(self.ship_details_frame, text="Max Speed: 20.0 knots")
        self.speed_label.pack(anchor=tk.W, pady=2)
        
        self.fuel_label = ttk.Label(self.ship_details_frame, text="Fuel Efficiency: 0.8")
        self.fuel_label.pack(anchor=tk.W, pady=2)
        
        self.cargo_label = ttk.Label(self.ship_details_frame, text="Cargo Capacity: 5,000 tons")
        self.cargo_label.pack(anchor=tk.W, pady=2)
        
        # Update ship details initially
        self.update_ship_details()
        
        # Route Objectives Frame
        objectives_frame = ttk.LabelFrame(input_frame, text="Route Objectives", padding=(5, 5, 5, 5))
        objectives_frame.pack(fill=tk.X, pady=10)
        
        # Sliders for objectives
        self.time_var = tk.DoubleVar(value=self.optimizer.objectives[RouteObjective.TIME])
        ttk.Label(objectives_frame, text="Time Priority:").pack(anchor=tk.W, pady=2)
        ttk.Scale(objectives_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.time_var, command=self.update_objectives).pack(fill=tk.X, pady=2)
        
        self.cost_var = tk.DoubleVar(value=self.optimizer.objectives[RouteObjective.COST])
        ttk.Label(objectives_frame, text="Cost Priority:").pack(anchor=tk.W, pady=2)
        ttk.Scale(objectives_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.cost_var, command=self.update_objectives).pack(fill=tk.X, pady=2)
        
        self.safety_var = tk.DoubleVar(value=self.optimizer.objectives[RouteObjective.SAFETY])
        ttk.Label(objectives_frame, text="Safety Priority:").pack(anchor=tk.W, pady=2)
        ttk.Scale(objectives_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                 variable=self.safety_var, command=self.update_objectives).pack(fill=tk.X, pady=2)
        
        # Advanced Options
        adv_options = ttk.LabelFrame(input_frame, text="Advanced Options", padding=(5, 5, 5, 5))
        adv_options.pack(fill=tk.X, pady=10)
        
        # Population size
        pop_frame = ttk.Frame(adv_options)
        pop_frame.pack(fill=tk.X, pady=3)
        ttk.Label(pop_frame, text="Population Size:").pack(side=tk.LEFT, padx=5)
        self.pop_size_var = tk.IntVar(value=self.optimizer.population_size)
        ttk.Spinbox(pop_frame, from_=50, to=500, increment=50, width=5, 
                   textvariable=self.pop_size_var).pack(side=tk.RIGHT, padx=5)
        
        # Generations
        gen_frame = ttk.Frame(adv_options)
        gen_frame.pack(fill=tk.X, pady=3)
        ttk.Label(gen_frame, text="Max Generations:").pack(side=tk.LEFT, padx=5)
        self.max_gen_var = tk.IntVar(value=self.optimizer.max_generations)
        ttk.Spinbox(gen_frame, from_=10, to=200, increment=10, width=5, 
                   textvariable=self.max_gen_var).pack(side=tk.RIGHT, padx=5)
        
        # Optimize Route Button
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.optimize_button = ttk.Button(
            button_frame, 
            text="Optimize Route", 
            command=self.optimize_route,
            style="Green.TButton"
        )
        self.optimize_button.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            button_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Save and load buttons
        btn_frame = ttk.Frame(button_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.save_button = ttk.Button(
            btn_frame, 
            text="Save Route", 
            command=self.save_route,
            style="Blue.TButton"
        )
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.load_button = ttk.Button(
            btn_frame, 
            text="Load Route", 
            command=self.load_route,
            style="Blue.TButton"
        )
        self.load_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
    
    def create_dashboard_section(self):
        """Create the dashboard section with metrics and charts"""
        # Dashboard Notebook
        self.dashboard_notebook = ttk.Notebook(self.right_column)
        self.dashboard_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab 1: Overview
        overview_tab = ttk.Frame(self.dashboard_notebook, padding=10)
        self.dashboard_notebook.add(overview_tab, text="Overview")
        
        # Upper section for metrics
        metrics_frame = ttk.LabelFrame(overview_tab, text="Route Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Metrics grid
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X, pady=5)
        
        # Metrics to track
        metrics = [
            "Total Distance (km)",
            "Estimated Time (hours)",
            "Fuel Cost ($)",
            "Safety Score",
            "Waypoint Count",
            "Average Speed (knots)"
        ]
        
        # Create variables for metrics
        self.metric_vars = {metric: tk.StringVar(value="N/A") for metric in metrics}
        
        # Display metrics in a 3x2 grid
        for i, (metric, var) in enumerate(self.metric_vars.items()):
            row, col = divmod(i, 3)
            
            metric_frame = ttk.Frame(metrics_grid, padding=5)
            metric_frame.grid(row=row, column=col, padx=10, pady=5, sticky=tk.W+tk.E)
            
            ttk.Label(metric_frame, text=metric + ":", font=("Segoe UI", 9)).pack(anchor=tk.W)
            metric_value = ttk.Label(
                metric_frame, 
                textvariable=var, 
                font=("Segoe UI", 14, "bold"), 
                foreground="#2980b9"
            )
            metric_value.pack(anchor=tk.W, pady=5)
        
        # Charts frame
        charts_frame = ttk.Frame(overview_tab)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left chart - Risk breakdown
        self.risk_chart_frame = ttk.LabelFrame(charts_frame, text="Risk Analysis")
        self.risk_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Right chart - Distance/Time
        self.distance_chart_frame = ttk.LabelFrame(charts_frame, text="Distance & Time")
        self.distance_chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Tab 2: Route Details
        details_tab = ttk.Frame(self.dashboard_notebook, padding=10)
        self.dashboard_notebook.add(details_tab, text="Route Details")
        
        # Route table frame
        route_table_frame = ttk.LabelFrame(details_tab, text="Waypoints", padding=10)
        route_table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create Treeview for waypoints
        columns = ("Port", "Latitude", "Longitude", "Weather Risk", "Piracy Risk", "Traffic", "Distance")
        self.waypoint_tree = ttk.Treeview(route_table_frame, columns=columns, show="headings", height=10)
        
        # Define headings
        for col in columns:
            self.waypoint_tree.heading(col, text=col)
            width = 100 if col in ("Port", "Distance") else 80
            self.waypoint_tree.column(col, width=width, anchor=tk.CENTER)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(route_table_frame, orient="vertical", command=self.waypoint_tree.yview)
        self.waypoint_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.waypoint_tree.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Map View (Placeholder)
    def create_map_view(self):
        """Create the map view tab in the dashboard notebook"""
        # Tab 3: Map View
        map_tab = ttk.Frame(self.dashboard_notebook, padding=10)
        self.dashboard_notebook.add(map_tab, text="Map View")
        
        # Map controls frame
        map_controls = ttk.Frame(map_tab)
        map_controls.pack(fill=tk.X, pady=5)
        
        # View map button
        self.view_map_btn = ttk.Button(
            map_controls,
            text="Open Map in Browser",
            command=self.open_map_in_browser,
            style="Orange.TButton"
        )
        self.view_map_btn.pack(side=tk.LEFT, padx=5)
        
        # Export options
        self.export_btn = ttk.Button(
            map_controls,
            text="Export Route Data",
            command=self.export_route_data,
            style="Blue.TButton"
        )
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Map preview frame
        self.map_preview_frame = ttk.LabelFrame(map_tab, text="Map Preview")
        self.map_preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Placeholder for map (will be replaced when a route is generated)
        self.map_placeholder = ttk.Label(
            self.map_preview_frame,
            text="Route map will be displayed here after optimization",
            font=("Segoe UI", 12),
            foreground="#95a5a6"
        )
        self.map_placeholder.pack(expand=True, pady=100)
        
        # Integration with folium
        self.map_html_path = None
    
    def update_ship_details(self):
        """Update ship details based on selected ship type"""
        ship_type = self.ship_type_var.get()
        ship = self.ship_params[ship_type]
        
        self.speed_label.config(text=f"Max Speed: {ship.max_speed} knots")
        self.fuel_label.config(text=f"Fuel Efficiency: {ship.fuel_efficiency}")
        self.cargo_label.config(text=f"Cargo Capacity: {ship.cargo_capacity:,.0f} tons")
    
    def update_objectives(self, *args):
        """Update optimization objectives based on slider values"""
        # Normalize to make sure they sum to 1.0
        total = self.time_var.get() + self.cost_var.get() + self.safety_var.get()
        
        self.optimizer.objectives[RouteObjective.TIME] = self.time_var.get() / total
        self.optimizer.objectives[RouteObjective.COST] = self.cost_var.get() / total
        self.optimizer.objectives[RouteObjective.SAFETY] = self.safety_var.get() / total
    
    def update_progress(self, value):
        """Update progress bar during optimization"""
        self.progress_var.set(value)
        self.master.update_idletasks()
    
    def optimize_route(self):
        """Run the route optimization process"""
        # Validate inputs
        if not self.start_port_var.get() or not self.end_port_var.get():
            messagebox.showerror("Error", "Please select start and end ports")
            return
            
        if self.start_port_var.get() == self.end_port_var.get():
            messagebox.showerror("Error", "Start and end ports must be different")
            return
        
        # Update optimizer parameters
        self.optimizer.population_size = self.pop_size_var.get()
        self.optimizer.max_generations = self.max_gen_var.get()
        
        # Find waypoint objects
        start_port = next(port for port in self.optimizer.ports if port.name == self.start_port_var.get())
        end_port = next(port for port in self.optimizer.ports if port.name == self.end_port_var.get())
        
        # Get ship parameters
        self.current_ship = self.ship_params[self.ship_type_var.get()]
        
        # Disable the optimize button during optimization
        self.optimize_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        try:
            # Show a busy cursor
            self.master.config(cursor="wait")
            self.master.update()
            
            # Optimize route with progress callback
            optimized_route = self.optimizer.optimize_route(
                start_port, 
                end_port, 
                self.current_ship,
                self.update_progress
            )
            
            # Set progress to 100%
            self.progress_var.set(100)
            
            # Calculate route fitness
            fitness = self.optimizer.calculate_route_fitness(optimized_route, self.current_ship)
            
            # Store current route and fitness
            self.current_route = optimized_route
            self.current_fitness = fitness
            
            # Update dashboard
            self.update_dashboard(optimized_route, fitness)
            
            # Generate map
            self.generate_map(optimized_route)
            
            # Show success message
            messagebox.showinfo("Success", "Route optimization completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during optimization: {str(e)}")
            logging.error(f"Optimization error: {str(e)}", exc_info=True)
        finally:
            # Re-enable the optimize button
            self.optimize_button.config(state=tk.NORMAL)
            # Restore cursor
            self.master.config(cursor="")
    
    def update_dashboard(self, route, fitness):
        """Update all dashboard elements with new route data"""
        # Calculate total distance
        def haversine_distance(wp1, wp2):
            R = 6371
            lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
            lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        total_distance = sum(haversine_distance(route[i], route[i+1]) for i in range(len(route)-1))
        
        # Update Metrics
        self.metric_vars["Total Distance (km)"].set(f"{total_distance:.1f}")
        self.metric_vars["Estimated Time (hours)"].set(f"{fitness[RouteObjective.TIME]:.1f}")
        self.metric_vars["Fuel Cost ($)"].set(f"{fitness[RouteObjective.COST]:.2f}")
        self.metric_vars["Safety Score"].set(f"{(1 - fitness[RouteObjective.SAFETY])*100:.1f}%")
        self.metric_vars["Waypoint Count"].set(f"{len(route)}")
        self.metric_vars["Average Speed (knots)"].set(f"{self.current_ship.max_speed:.1f}")
        
        # Update route details table
        self.update_route_table(route)
        
        # Update charts
        self.update_risk_chart(route)
        self.update_distance_chart(route)
    
    def update_route_table(self, route):
        """Update the route details table with waypoint information"""
        # Clear existing data
        for item in self.waypoint_tree.get_children():
            self.waypoint_tree.delete(item)
        
        # Calculate distances between waypoints
        def haversine_distance(wp1, wp2):
            R = 6371
            lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
            lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        # Add waypoints to table
        for i, wp in enumerate(route):
            distance = "-"
            if i < len(route) - 1:
                distance = f"{haversine_distance(wp, route[i+1]):.1f} km"
            
            self.waypoint_tree.insert(
                "", 
                "end", 
                values=(
                    wp.name,
                    f"{wp.latitude:.4f}",
                    f"{wp.longitude:.4f}",
                    f"{wp.weather_risk:.2f}",
                    f"{wp.piracy_risk:.2f}",
                    f"{wp.maritime_traffic:.2f}",
                    distance
                )
            )
    
    def update_risk_chart(self, route):
        """Update risk analysis chart"""
        # Clear existing chart
        for widget in self.risk_chart_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        
        # Extract risk data
        ports = [wp.name for wp in route]
        weather_risks = [wp.weather_risk for wp in route]
        piracy_risks = [wp.piracy_risk for wp in route]
        traffic_risks = [wp.maritime_traffic for wp in route]
        
        # Create stacked bar chart
        width = 0.6
        ind = np.arange(len(ports))
        
        ax.bar(ind, weather_risks, width, label='Weather Risk', color='#3498db')
        ax.bar(ind, piracy_risks, width, bottom=weather_risks, label='Piracy Risk', color='#e74c3c')
        ax.bar(ind, traffic_risks, width, bottom=np.array(weather_risks) + np.array(piracy_risks), 
              label='Traffic Risk', color='#f39c12')
        
        # Add labels and legend
        ax.set_ylabel('Risk Level')
        ax.set_title('Risk Analysis by Port')
        ax.set_xticks(ind)
        ax.set_xticklabels(ports, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.risk_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_distance_chart(self, route):
        """Update distance and time chart"""
        # Clear existing chart
        for widget in self.distance_chart_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        
        # Calculate leg distances
        def haversine_distance(wp1, wp2):
            R = 6371
            lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
            lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        # Get leg names and distances
        leg_names = []
        distances = []
        
        for i in range(len(route)-1):
            leg_names.append(f"{route[i].name}-{route[i+1].name}")
            distances.append(haversine_distance(route[i], route[i+1]))
        
        # Create bar chart
        ax.bar(leg_names, distances, color='#2980b9')
        
        # Add a second y-axis for time
        ax2 = ax.twinx()
        times = [d / self.current_ship.max_speed for d in distances]
        ax2.plot(leg_names, times, 'ro-', linewidth=2, markersize=8)
        
        # Add labels
        ax.set_ylabel('Distance (km)', color='#2980b9')
        ax2.set_ylabel('Time (hours)', color='red')
        ax.set_title('Distance and Time by Route Leg')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add legend
        distance_patch = mpatches.Patch(color='#2980b9', label='Distance')
        time_patch = mpatches.Patch(color='red', label='Time')
        ax.legend(handles=[distance_patch, time_patch], loc='upper left')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.distance_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_map(self, route):
        """Generate the interactive map for the route"""
        # Generate a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"route_map_{timestamp}.html"
        
        # Create the map
        display_route_on_map(route, filename=filename, show_browser=False)
        
        # Store the path
        self.map_html_path = os.path.realpath(filename)
        
        # Update the map preview placeholder with a message
        self.map_placeholder.config(
            text=f"Map generated successfully! Click 'Open Map in Browser' to view interactive map.",
            foreground="#27ae60"
        )
    
    def open_map_in_browser(self):
        """Open the generated map in a browser"""
        if self.map_html_path and os.path.exists(self.map_html_path):
            webbrowser.open('file://' + self.map_html_path)
        else:
            messagebox.showerror("Error", "No map has been generated yet. Please optimize a route first.")
    
    def export_route_data(self):
        """Export the route data to a JSON file"""
        if not self.current_route:
            messagebox.showerror("Error", "No route has been generated yet. Please optimize a route first.")
            return
        
        try:
            # Prepare route data
            route_data = {
                "route_info": {
                    "start_port": self.current_route[0].name,
                    "end_port": self.current_route[-1].name,
                    "ship_type": self.current_ship.ship_type,
                    "optimization_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "metrics": {
                    "total_distance_km": sum(self.haversine_distance(self.current_route[i], self.current_route[i+1]) 
                                           for i in range(len(self.current_route)-1)),
                    "estimated_time_hours": self.current_fitness[RouteObjective.TIME],
                    "fuel_cost": self.current_fitness[RouteObjective.COST],
                    "safety_score": 1 - self.current_fitness[RouteObjective.SAFETY]
                },
                "waypoints": []
            }
            
            # Add waypoint details
            for wp in self.current_route:
                route_data["waypoints"].append({
                    "name": wp.name,
                    "latitude": wp.latitude,
                    "longitude": wp.longitude,
                    "weather_risk": wp.weather_risk,
                    "piracy_risk": wp.piracy_risk,
                    "maritime_traffic": wp.maritime_traffic
                })
            
            # Ask user for save location
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Route Data"
            )
            
            if filename:
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(route_data, f, indent=4)
                
                messagebox.showinfo("Success", f"Route data exported successfully to {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export route data: {str(e)}")
    
    def haversine_distance(self, wp1, wp2):
        """Calculate distance between two waypoints using Haversine formula"""
        R = 6371
        lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
        lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def save_route(self):
        """Save the current route configuration"""
        if not self.current_route:
            messagebox.showerror("Error", "No route has been generated yet. Please optimize a route first.")
            return
        
        try:
            # Ask user for save location
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".mro",
                filetypes=[("Maritime Route files", "*.mro"), ("All files", "*.*")],
                title="Save Route Configuration"
            )
            
            if filename:
                # Prepare data to save
                save_data = {
                    "start_port": self.start_port_var.get(),
                    "end_port": self.end_port_var.get(),
                    "ship_type": self.ship_type_var.get(),
                    "objectives": {
                        "time": self.time_var.get(),
                        "cost": self.cost_var.get(),
                        "safety": self.safety_var.get()
                    },
                    "advanced_options": {
                        "population_size": self.pop_size_var.get(),
                        "max_generations": self.max_gen_var.get()
                    }
                }
                
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(save_data, f, indent=4)
                
                messagebox.showinfo("Success", "Route configuration saved successfully!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save route configuration: {str(e)}")
    
    def load_route(self):
        """Load a saved route configuration"""
        try:
            # Ask user for file location
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                filetypes=[("Maritime Route files", "*.mro"), ("All files", "*.*")],
                title="Load Route Configuration"
            )
            
            if filename and os.path.exists(filename):
                # Load data from file
                with open(filename, 'r') as f:
                    load_data = json.load(f)
                
                # Apply settings
                self.start_port_var.set(load_data["start_port"])
                self.end_port_var.set(load_data["end_port"])
                self.ship_type_var.set(load_data["ship_type"])
                
                # Set objectives
                self.time_var.set(load_data["objectives"]["time"])
                self.cost_var.set(load_data["objectives"]["cost"])
                self.safety_var.set(load_data["objectives"]["safety"])
                self.update_objectives()
                
                # Set advanced options
                self.pop_size_var.set(load_data["advanced_options"]["population_size"])
                self.max_gen_var.set(load_data["advanced_options"]["max_generations"])
                
                # Update ship details
                self.update_ship_details()
                
                messagebox.showinfo("Success", "Route configuration loaded successfully!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load route configuration: {str(e)}")

# Main function
def main():
    try:
        # Try to use ttkthemes if available
        root = ThemedTk(theme="arc")
    except:
        # Fallback to regular Tk
        root = tk.Tk()
        print("ttkthemes not available. Using default Tk theme.")
    
    app = ModernMaritimeRouteOptimizerApp(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    root.mainloop()

if __name__ == "__main__":
    main()    