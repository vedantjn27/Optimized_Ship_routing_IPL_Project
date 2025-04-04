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
            a = np.sin(dlat/2)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)*2
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
                       ship: ShipParameters) -> List[Waypoint]:
        population = self.generate_initial_population(start, end)
        for generation in range(self.max_generations):
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

def display_route_on_map(route: List[Waypoint]):
    map_route = folium.Map(location=[route[0].latitude, route[0].longitude], zoom_start=4)
    for wp in route:
        folium.Marker(
            location=[wp.latitude, wp.longitude],
            popup=f"{wp.name}\nWeather Risk: {wp.weather_risk:.2f}\nPiracy Risk: {wp.piracy_risk:.2f}",
            icon=folium.Icon(color="blue")
        ).add_to(map_route)
    folium.PolyLine(
        locations=[[wp.latitude, wp.longitude] for wp in route],
        color="red",
        weight=3
    ).add_to(map_route)
    map_route.save("optimized_route_map.html")

class MaritimeRouteOptimizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Maritime Route Optimizer")
        master.geometry("1200x800")
        master.configure(bg='#f0f0f0')

        # Create main container
        self.main_container = ttk.Frame(master, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Optimizer instance
        self.optimizer = MaritimeRouteOptimizer()

        # Input Section
        self.create_input_section()

        # Dashboard Section
        self.create_dashboard_section()

    def create_input_section(self):
        input_frame = ttk.LabelFrame(self.main_container, text="Route Configuration")
        input_frame.pack(fill=tk.X, pady=10)

        # Port names extracted from self.optimizer.ports
        ports = [port.name for port in self.optimizer.ports]

        # Start Port Selection
        ttk.Label(input_frame, text="Start Port:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_port_var = tk.StringVar()
        self.start_port_dropdown = ttk.Combobox(input_frame, textvariable=self.start_port_var, values=ports)
        self.start_port_dropdown.grid(row=0, column=1, padx=5, pady=5)

        # End Port Selection
        ttk.Label(input_frame, text="End Port:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.end_port_var = tk.StringVar()
        self.end_port_dropdown = ttk.Combobox(input_frame, textvariable=self.end_port_var, values=ports)
        self.end_port_dropdown.grid(row=1, column=1, padx=5, pady=5)

        # Ship Type Selection
        ship_types = ["Cargo", "Container", "Tanker", "Passenger"]
        ttk.Label(input_frame, text="Ship Type:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.ship_type_var = tk.StringVar(value="Cargo")
        self.ship_type_dropdown = ttk.Combobox(input_frame, textvariable=self.ship_type_var, values=ship_types)
        self.ship_type_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Optimize Route Button
        self.optimize_button = ttk.Button(input_frame, text="Optimize Route", command=self.optimize_route)
        self.optimize_button.grid(row=3, column=0, columnspan=2, pady=10)

    def create_dashboard_section(self):
        dashboard_frame = ttk.LabelFrame(self.main_container, text="Route Dashboard")
        dashboard_frame.pack(fill=tk.X, expand=True, pady=10)

        # Metrics to track
        metrics = [
            "Total Distance",
            "Route Waypoints",
            "Estimated Time",
            "Fuel Consumption",
            "Safety Index"
        ]

        # Create variables for metrics
        self.metric_vars = {metric: tk.StringVar(value="N/A") for metric in metrics}

        # Display metrics
        for i, (metric, var) in enumerate(self.metric_vars.items()):
            ttk.Label(dashboard_frame, text=f"{metric}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Label(dashboard_frame, textvariable=var).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)

        # Text widget for route details
        self.route_details_text = tk.Text(dashboard_frame, height=10, width=50, wrap=tk.WORD)
        self.route_details_text.grid(row=len(metrics), column=0, columnspan=2, padx=5, pady=5)

    def optimize_route(self):
        # Validate inputs
        if not self.start_port_var.get() or not self.end_port_var.get():
            messagebox.showerror("Error", "Please select start and end ports")
            return

        # Find waypoint objects
        start_port = next(port for port in self.optimizer.ports if port.name == self.start_port_var.get())
        end_port = next(port for port in self.optimizer.ports if port.name == self.end_port_var.get())

        # Ship parameters based on type
        ship_params = {
            "Cargo": ShipParameters(ship_type="Cargo", fuel_efficiency=0.8, max_speed=20.0, cargo_capacity=5000.0),
            "Container": ShipParameters(ship_type="Container", fuel_efficiency=0.7, max_speed=18.0, cargo_capacity=8000.0),
            "Tanker": ShipParameters(ship_type="Tanker", fuel_efficiency=0.6, max_speed=15.0, cargo_capacity=10000.0),
            "Passenger": ShipParameters(ship_type="Passenger", fuel_efficiency=0.5, max_speed=22.0, cargo_capacity=2000.0)
        }

        # Optimize route
        optimized_route = self.optimizer.optimize_route(start_port, end_port, ship_params[self.ship_type_var.get()])
        
        # Calculate route fitness
        fitness = self.optimizer.calculate_route_fitness(optimized_route, ship_params[self.ship_type_var.get()])

        # Update dashboard
        self.update_dashboard(optimized_route, fitness)

        # Generate map
        display_route_on_map(optimized_route)
        webbrowser.open('file://' + os.path.realpath('optimized_route_map.html'))

    def update_dashboard(self, route, fitness):
        # Total Distance Calculation
        def haversine_distance(wp1, wp2):
            R = 6371
            lat1, lon1 = np.radians(wp1.latitude), np.radians(wp1.longitude)
            lat2, lon2 = np.radians(wp2.latitude), np.radians(wp2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)*2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        total_distance = sum(haversine_distance(route[i], route[i+1]) for i in range(len(route)-1))

        # Update Metrics
        self.metric_vars["Total Distance"].set(f"{total_distance:.2f} km")
        self.metric_vars["Route Waypoints"].set(", ".join(wp.name for wp in route))
        self.metric_vars["Estimated Time"].set(f"{fitness[RouteObjective.TIME]:.2f} hours")
        self.metric_vars["Fuel Consumption"].set(f"${fitness[RouteObjective.COST]:.2f}")
        self.metric_vars["Safety Index"].set(f"{(1 - fitness[RouteObjective.SAFETY])*100:.2f}%")

        # Clear and update route details
        self.route_details_text.delete(1.0, tk.END)
        route_info = "Optimized Route Details:\n\n"
        for i, wp in enumerate(route):
            route_info += f"{i+1}. {wp.name}\n"
            route_info += f"   Latitude: {wp.latitude}, Longitude: {wp.longitude}\n"
            route_info += f"   Weather Risk: {wp.weather_risk:.2f}\n"
            route_info += f"   Piracy Risk: {wp.piracy_risk:.2f}\n"
            route_info += f"   Maritime Traffic: {wp.maritime_traffic:.2f}\n\n"
        
        self.route_details_text.insert(tk.END, route_info)

def main():
    root = tk.Tk()
    app = MaritimeRouteOptimizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()