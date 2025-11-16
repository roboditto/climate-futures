"""
Hydrological Flood Simulation Engine - Days 7-8
Simulates water flow and accumulation using terrain data.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt


class FloodSimulator:
    """
    Simulates flood propagation using cellular automata and terrain flow.
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize flood simulator.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        logger : logging.Logger, optional
            Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.sim_config = config.get('flood_simulation', {})
        self.resolution = self.sim_config.get('dem_resolution', 30)
        self.manning_n = self.sim_config.get('manning_coefficient', 0.035)
        self.timestep = self.sim_config.get('timestep', 300)
        self.max_iterations = self.sim_config.get('max_iterations', 1000)
    
    def generate_synthetic_dem(self, size: Tuple[int, int] = (100, 100),
                              elevation_range: Tuple[float, float] = (0, 100)) -> np.ndarray:
        """
        Generate synthetic Digital Elevation Model.
        
        Parameters:
        -----------
        size : tuple
            Grid size (rows, cols)
        elevation_range : tuple
            (min_elevation, max_elevation) in meters
        
        Returns:
        --------
        np.ndarray
            2D elevation grid
        """
        self.logger.info(f"Generating synthetic DEM: {size[0]}x{size[1]}")
        
        rows, cols = size
        min_elev, max_elev = elevation_range
        
        # Generate base terrain using Perlin-like noise
        np.random.seed(42)
        
        # Create multiple frequency components
        dem = np.zeros((rows, cols))
        
        # Large-scale features (hills/valleys)
        x = np.linspace(0, 4*np.pi, cols)
        y = np.linspace(0, 4*np.pi, rows)
        X, Y = np.meshgrid(x, y)
        
        dem += 30 * np.sin(X) * np.cos(Y)
        dem += 20 * np.cos(2*X) * np.sin(2*Y)
        dem += 15 * np.sin(3*X + 2*Y)
        
        # Add smaller features
        for scale in [10, 5, 2]:
            noise = np.random.randn(rows, cols) * scale
            dem += ndimage.gaussian_filter(noise, sigma=3)
        
        # Normalize to elevation range
        dem = (dem - dem.min()) / (dem.max() - dem.min())
        dem = dem * (max_elev - min_elev) + min_elev
        
        # Add coastal gradient (lower on one side)
        coastal_gradient = np.linspace(1.0, 0.3, cols)
        dem = dem * coastal_gradient
        
        # Smooth terrain
        dem = ndimage.gaussian_filter(dem, sigma=2)
        
        return dem
    
    def calculate_flow_direction_d8(self, dem: np.ndarray) -> np.ndarray:
        """
        Calculate flow direction using D8 algorithm.
        
        Each cell flows to its steepest downhill neighbor.
        
        Parameters:
        -----------
        dem : np.ndarray
            Digital Elevation Model
        
        Returns:
        --------
        np.ndarray
            Flow direction grid (0-7 for 8 directions, -1 for sinks)
        """
        self.logger.info("Calculating D8 flow directions...")
        
        rows, cols = dem.shape
        flow_dir = np.full((rows, cols), -1, dtype=np.int8)
        
        # 8 directions: [N, NE, E, SE, S, SW, W, NW]
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                current_elev = dem[i, j]
                max_slope = 0
                max_dir = -1
                
                for d, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj
                    neighbor_elev = dem[ni, nj]
                    
                    # Calculate slope
                    distance = np.sqrt(di**2 + dj**2) * self.resolution
                    slope = (current_elev - neighbor_elev) / distance
                    
                    if slope > max_slope:
                        max_slope = slope
                        max_dir = d
                
                flow_dir[i, j] = max_dir
        
        return flow_dir
    
    def calculate_flow_accumulation(self, flow_dir: np.ndarray) -> np.ndarray:
        """
        Calculate flow accumulation (how many cells drain to each cell).
        
        Parameters:
        -----------
        flow_dir : np.ndarray
            Flow direction grid
        
        Returns:
        --------
        np.ndarray
            Flow accumulation grid
        """
        self.logger.info("Calculating flow accumulation...")
        
        rows, cols = flow_dir.shape
        accumulation = np.ones((rows, cols), dtype=np.float32)
        
        # Directions mapping
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        # Sort cells by elevation (process from high to low)
        # Simplified: iterate multiple times
        for iteration in range(rows + cols):
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    direction = flow_dir[i, j]
                    
                    if direction >= 0:
                        di, dj = directions[direction]
                        ni, nj = i + di, j + dj
                        
                        if 0 <= ni < rows and 0 <= nj < cols:
                            accumulation[ni, nj] += accumulation[i, j]
        
        return accumulation
    
    def simulate_rainfall_runoff(self, dem: np.ndarray,
                                 rainfall_mm: float,
                                 duration_hours: float = 1.0) -> np.ndarray:
        """
        Simulate rainfall-runoff process.
        
        Parameters:
        -----------
        dem : np.ndarray
            Digital Elevation Model
        rainfall_mm : float
            Rainfall amount in millimeters
        duration_hours : float
            Rainfall duration in hours
        
        Returns:
        --------
        np.ndarray
            Water depth grid (meters)
        """
        self.logger.info(f"Simulating {rainfall_mm} mm rainfall over {duration_hours} hours")
        
        rows, cols = dem.shape
        
        # Convert rainfall to meters
        rainfall_m = rainfall_mm / 1000.0
        
        # Initialize water depth
        water_depth = np.full((rows, cols), rainfall_m, dtype=np.float32)
        
        # Calculate slope
        slope = self.calculate_slope(dem)
        
        # Get flow directions
        flow_dir = self.calculate_flow_direction_d8(dem)
        
        # Directions mapping
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        # Simulate water redistribution
        num_steps = min(50, self.max_iterations)
        
        for step in range(num_steps):
            new_water = water_depth.copy()
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if water_depth[i, j] > 0.001:  # Only process cells with water
                        direction = flow_dir[i, j]
                        
                        if direction >= 0:
                            di, dj = directions[direction]
                            ni, nj = i + di, j + dj
                            
                            if 0 <= ni < rows and 0 <= nj < cols:
                                # Calculate flow velocity (Manning's equation simplified)
                                velocity = (1.0 / self.manning_n) * (slope[i, j] ** 0.5) * (water_depth[i, j] ** 0.67)
                                
                                # Calculate flow volume
                                flow_volume = velocity * water_depth[i, j] * self.timestep / duration_hours
                                flow_volume = min(flow_volume, water_depth[i, j] * 0.5)  # Max 50% per step
                                
                                # Transfer water
                                new_water[i, j] -= flow_volume
                                new_water[ni, nj] += flow_volume
            
            water_depth = new_water
            
            # Evaporation/infiltration (simplified)
            water_depth *= 0.99
        
        self.logger.info(f"Simulation complete. Max depth: {water_depth.max():.3f} m")
        
        return water_depth
    
    def calculate_slope(self, dem: np.ndarray) -> np.ndarray:
        """
        Calculate slope from DEM.
        
        Parameters:
        -----------
        dem : np.ndarray
            Digital Elevation Model
        
        Returns:
        --------
        np.ndarray
            Slope grid (in radians)
        """
        # Calculate gradient
        dy, dx = np.gradient(dem)
        
        # Calculate slope magnitude
        slope = np.sqrt(dx**2 + dy**2) / self.resolution
        
        # Convert to angle (avoid division by zero)
        slope = np.arctan(slope)
        
        # Ensure minimum slope
        slope = np.maximum(slope, 0.001)
        
        return slope
    
    def identify_flood_zones(self, water_depth: np.ndarray,
                            threshold_m: float = 0.1) -> np.ndarray:
        """
        Identify areas with significant flooding.
        
        Parameters:
        -----------
        water_depth : np.ndarray
            Water depth grid
        threshold_m : float
            Minimum depth to consider as flooded (meters)
        
        Returns:
        --------
        np.ndarray
            Binary flood zone mask
        """
        flood_zones = (water_depth > threshold_m).astype(int)
        
        flooded_area = flood_zones.sum() * (self.resolution ** 2) / 10000  # hectares
        self.logger.info(f"Flooded area: {flooded_area:.2f} hectares")
        
        return flood_zones
    
    def plot_simulation(self, dem: np.ndarray,
                       water_depth: np.ndarray,
                       save_path: Optional[str] = None) -> None:
        """
        Visualize flood simulation results.
        
        Parameters:
        -----------
        dem : np.ndarray
            Digital Elevation Model
        water_depth : np.ndarray
            Water depth grid
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # DEM
        im1 = axes[0].imshow(dem, cmap='terrain', aspect='auto')
        axes[0].set_title('Digital Elevation Model')
        axes[0].set_xlabel('X (grid cells)')
        axes[0].set_ylabel('Y (grid cells)')
        plt.colorbar(im1, ax=axes[0], label='Elevation (m)')
        
        # Water depth
        im2 = axes[1].imshow(water_depth, cmap='Blues', aspect='auto')
        axes[1].set_title('Water Depth')
        axes[1].set_xlabel('X (grid cells)')
        axes[1].set_ylabel('Y (grid cells)')
        plt.colorbar(im2, ax=axes[1], label='Depth (m)')
        
        # Flood zones overlay
        flood_zones = self.identify_flood_zones(water_depth, threshold_m=0.05)
        axes[2].imshow(dem, cmap='terrain', alpha=0.7, aspect='auto')
        axes[2].imshow(flood_zones, cmap='Blues', alpha=0.5, aspect='auto')
        axes[2].set_title('Flood Risk Zones')
        axes[2].set_xlabel('X (grid cells)')
        axes[2].set_ylabel('Y (grid cells)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Simulation plot saved to {save_path}")
        
        plt.show()


def main():
    """Example usage."""
    from utils import load_config, setup_logging
    
    config = load_config()
    logger = setup_logging(config)
    
    # Initialize simulator
    simulator = FloodSimulator(config, logger)
    
    # Generate DEM
    print("Generating terrain...")
    dem = simulator.generate_synthetic_dem(size=(150, 150), elevation_range=(0, 100))
    
    # Simulate heavy rainfall
    print("Simulating flood from 100mm rainfall...")
    water_depth = simulator.simulate_rainfall_runoff(dem, rainfall_mm=100, duration_hours=6)
    
    # Visualize
    print("Creating visualization...")
    simulator.plot_simulation(dem, water_depth, save_path='results/flood_simulation.png')
    
    print("\nSimulation complete!")
    print(f"Maximum water depth: {water_depth.max():.3f} m")
    print(f"Average water depth: {water_depth.mean():.3f} m")


if __name__ == "__main__":
    main()
