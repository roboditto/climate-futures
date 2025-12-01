import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional
import logging
import matplotlib
# Use non-interactive backend for headless/scripted environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import datetime
import gzip
import requests
from typing import List

# Optional rasterio for real DEM handling
try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except Exception:
    # Ensure names exist even if rasterio is not installed so static
    # analysis and later guarded references won't see an unbound name.
    rasterio = None
    Resampling = None
    RASTERIO_AVAILABLE = False

# Optional mapping visualization
try:
    import folium
    from folium.raster_layers import ImageOverlay
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


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

    def load_dem_from_raster(self, path: str, target_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Load a DEM from a raster file and optionally resample to target shape.

        Returns (dem_array, bounds) where bounds = (south, west, north, east).
        """
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio is required to load real DEMs")

        with rasterio.open(path) as src:
            # Read first band
            if target_shape is None:
                arr = src.read(1)
            else:
                out_shape = (int(target_shape[0]), int(target_shape[1]))
                # rasterio expects (bands, rows, cols)
                arr = src.read(1, out_shape=(out_shape[0], out_shape[1]), resampling=Resampling.bilinear)

            bounds = src.bounds  # left, bottom, right, top (west, south, east, north)
            # convert to (south, west, north, east)
            south = bounds.bottom
            west = bounds.left
            north = bounds.top
            east = bounds.right

        # Mask out nodata if present
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr, (south, west, north, east)

    def generate_spatial_rainfall(self, dem: np.ndarray, total_mm: float = 100.0,
                                  pattern: str = 'orographic', n_hotspots: int = 3,
                                  seed: Optional[int] = 42) -> np.ndarray:
        """
        Generate a spatial rainfall field (mm) matching the DEM shape.

        Patterns:
          - 'orographic': more rain on windward (eastern) slopes and higher elevations
          - 'convective': random Gaussian convective cells added
        """
        np.random.seed(seed)
        rows, cols = dem.shape

        # Base uniform field
        base = np.ones((rows, cols), dtype=float)

        # Elevation influence (normalize elevation)
        elev = dem.copy().astype(float)
        elev = np.nan_to_num(elev, nan=0.0)
        if elev.max() - elev.min() > 0:
            elev_norm = (elev - elev.min()) / (elev.max() - elev.min())
        else:
            elev_norm = np.zeros_like(elev)

        # Eastness: columns with lower longitude index are west; assume wind from east -> higher rainfall on eastern columns
        x = np.linspace(0, 1, cols)
        east_factor = np.tile(x, (rows, 1))  # increases to the east

        field = base
        if pattern in ('orographic', 'both'):
            # amplify where elevation is higher and on eastern side
            field = field * (1.0 + 0.8 * elev_norm + 0.6 * east_factor)

        if pattern in ('convective', 'both'):
            # add several Gaussian hotspots
            yy, xx = np.indices((rows, cols))
            for _ in range(n_hotspots):
                cy = np.random.uniform(0, rows)
                cx = np.random.uniform(0, cols)
                sigma_y = np.random.uniform(rows*0.02, rows*0.15)
                sigma_x = np.random.uniform(cols*0.02, cols*0.15)
                amp = np.random.uniform(0.5, 2.0)
                gauss = amp * np.exp(-(((yy-cy)**2)/(2*sigma_y**2) + ((xx-cx)**2)/(2*sigma_x**2)))
                field += gauss

        # Normalize field to have mean 1, then scale to total_mm
        field_mean = field.mean()
        if field_mean <= 0:
            field_mean = 1.0
        scaled = field / field_mean * total_mm
        return scaled

    def fetch_srtm(self, bounds: Tuple[float, float, float, float], out_path: str = 'data/dem_jamaica.tif') -> Optional[str]:
        """
        Try to fetch SRTM data covering bounds and save to out_path using the `elevation` package.

        Returns out_path on success or None on failure.
        """
        try:
            import elevation
        except Exception:
            self.logger.info('elevation package not available; skipping SRTM fetch')
            return None

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        south, west, north, east = bounds
        try:
            self.logger.info('Downloading SRTM data with elevation.clip')
            elevation.clip(bounds=(west, south, east, north), output=out_path)
            self.logger.info(f'Saved SRTM to {out_path}')
            return out_path
        except Exception as e:
            self.logger.warning(f'Failed to fetch SRTM via elevation: {e}')
            return None

    def download_chirps_sum(self, start_date: str, end_date: str,
                            bounds: Tuple[float, float, float, float],
                            target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Download CHIRPS daily TIFFs for date range and sum precipitation into a raster clipped to bounds.

        start_date/end_date: 'YYYY-MM-DD'
        Returns numpy array (mm) or None on failure.
        """
        if not RASTERIO_AVAILABLE:
            self.logger.info('rasterio not available; cannot process CHIRPS files locally')
            return None

        # CHIRPS daily URL pattern (p05 resolution)
        def chirps_url(dt: datetime.date) -> str:
            return f'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{dt.year}/chirps-v2.0.{dt.year}.{dt.month:02d}.{dt.day:02d}.tif.gz'

        sd = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        ed = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        day = sd
        sum_arr = None
        downloaded = 0

        while day <= ed:
            url = chirps_url(day)
            try:
                self.logger.info(f'Downloading CHIRPS {day.isoformat()}')
                r = requests.get(url, timeout=30)
                if r.status_code != 200:
                    self.logger.debug(f'CHIRPS not found for {day} (HTTP {r.status_code})')
                    day += datetime.timedelta(days=1)
                    continue

                # Decompress
                buf = gzip.decompress(r.content)

                with rasterio.MemoryFile(buf) as memfile:
                    with memfile.open() as src:
                        arr = np.asarray(src.read(1), dtype=float)
                        # CHIRPS units are mm/day so use as-is
                        if sum_arr is None:
                            sum_arr = arr.copy()
                        else:
                            # Resample to sum_arr shape if different using ndimage.zoom
                            if sum_arr.shape != arr.shape:
                                zoom_factors = (sum_arr.shape[0] / arr.shape[0], sum_arr.shape[1] / arr.shape[1])
                                arr_resampled = ndimage.zoom(arr, zoom_factors, order=1)
                                sum_arr += arr_resampled
                            else:
                                sum_arr += arr
                downloaded += 1
            except Exception as e:
                self.logger.debug(f'Failed to download or process CHIRPS for {day}: {e}')

            day += datetime.timedelta(days=1)

        if sum_arr is None:
            self.logger.info('No CHIRPS files downloaded')
            return None

        # If a target_shape provided, resample to that
        if target_shape is not None and sum_arr.shape != target_shape:
            zoom_factors = (target_shape[0] / sum_arr.shape[0], target_shape[1] / sum_arr.shape[1])
            sum_arr = ndimage.zoom(sum_arr, zoom_factors, order=1)

        self.logger.info(f'Downloaded and summed CHIRPS for {downloaded} days')
        return sum_arr

    def map_grid_to_bounds(self, grid_shape: Tuple[int, int], bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Return bounds (south, west, north, east) for the grid to overlay on a map.

        Parameters:
        -----------
        grid_shape : tuple
            (rows, cols)
        bounds : tuple
            (south, west, north, east)

        Returns:
        --------
        tuple
            Same bounds (pass-through) – helper kept for clarity/extension.
        """
        return bounds
    
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
                                 rainfall_mm: float = 0.0,
                                 duration_hours: float = 1.0,
                                 rainfall_field: Optional[np.ndarray] = None) -> np.ndarray:
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
        
        # Initialize water depth either from a spatial rainfall_field (mm) or uniform rainfall_mm
        if rainfall_field is not None:
            if rainfall_field.shape != (rows, cols):
                # try to resample using simple interpolation
                self.logger.info('Resampling rainfall_field to DEM shape')
                rainfall_resampled = ndimage.zoom(rainfall_field, (rows / rainfall_field.shape[0], cols / rainfall_field.shape[1]), order=1)
            else:
                rainfall_resampled = rainfall_field

            rainfall_m = rainfall_resampled / 1000.0
            water_depth = rainfall_m.astype(np.float32)
        else:
            rainfall_m = rainfall_mm / 1000.0
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
                       save_path: Optional[str] = None,
                       country: Optional[str] = None) -> None:
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
        country : str, optional
            Country name to display in the title
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
        water_title = f'Water Depth - {country}' if country else 'Water Depth'
        axes[1].set_title(water_title)
        axes[1].set_xlabel('X (grid cells)')
        axes[1].set_ylabel('Y (grid cells)')
        plt.colorbar(im2, ax=axes[1], label='Depth (m)')
        
        # Flood zones overlay
        flood_zones = self.identify_flood_zones(water_depth, threshold_m=0.05)
        axes[2].imshow(dem, cmap='terrain', alpha=0.7, aspect='auto')
        axes[2].imshow(flood_zones, cmap='Blues', alpha=0.5, aspect='auto')
        flood_title = f'Flood Risk Zones - {country}' if country else 'Flood Risk Zones'
        axes[2].set_title(flood_title)
        axes[2].set_xlabel('X (grid cells)')
        axes[2].set_ylabel('Y (grid cells)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Simulation plot saved to {save_path}")
            plt.close()  # Close the figure instead of showing it
        else:
            plt.show()  # Only show if not saving

    def export_map(self, dem: np.ndarray, water_depth: np.ndarray,
                   save_html: str = 'results/flood_map.html',
                   bounds: Optional[Tuple[float, float, float, float]] = None,
                   cmap: str = 'Blues',
                   country: Optional[str] = None) -> None:
        """
        Export an interactive Folium map showing water depth overlayed on a basemap.

        Parameters:
        -----------
        dem : np.ndarray
            DEM grid
        water_depth : np.ndarray
            Water depth grid
        save_html : str
            Output HTML file path
        bounds : tuple, optional
            (south, west, north, east) bounds to place the overlay. If None,
            a default bounding box for Jamaica is used.
        cmap : str
            Matplotlib colormap for water depth
        country : str, optional
            Country name to display in the map title
        """
        if not FOLIUM_AVAILABLE:
            self.logger.warning('Folium not installed; skipping map export')
            return

        # Default bounds approx for Jamaica: south, west, north, east
        if bounds is None:
            bounds = (17.7, -78.4, 18.5, -76.1)

        # Normalize water depth to 0-1 for colormap
        wd = np.nan_to_num(water_depth, nan=0.0, posinf=0.0, neginf=0.0)
        vmax = max(wd.max(), 1e-6)
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
        cmap_mpl = matplotlib.colormaps.get_cmap(cmap)

        # Convert to RGBA image
        rgba = cmap_mpl(norm(wd))  # shape (rows, cols, 4)
        # Convert to 8-bit RGBA and flip vertically for correct georeferencing
        rgba_img = (rgba * 255).astype(np.uint8)
        rgba_img = np.flipud(rgba_img)

        img = Image.fromarray(rgba_img)

        # Create folium map centered on bounds
        south, west, north, east = bounds
        center_lat = (south + north) / 2.0
        center_lon = (west + east) / 2.0

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

        # Save image to temporary PNG file and overlay
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            image_overlay = ImageOverlay(image=tmp_path, bounds=[[south, west], [north, east]], opacity=0.6)
            image_overlay.add_to(m)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Add a legend marker for max depth and country name
        legend_text = f"Max depth: {vmax:.2f} m"
        if country:
            legend_text += f"<br/>Country: {country}"
        folium.map.Marker([north, west], icon=folium.DivIcon(html=f"<div style='font-size:12px'>{legend_text}</div>" )).add_to(m)

        # Save map
        m.save(save_html)
        map_title = f"Flood Map - {country}" if country else "Flood Map"
        self.logger.info(f"Exported interactive {map_title} to {save_html}")


def main(interactive: bool = False, use_srtm: bool = False, use_chirps: bool = False,
         chirps_start: str = '', chirps_end: str = ''):
    """Example usage with optional observational data sources."""
    from utils import load_config, setup_logging
    
    config = load_config()
    logger = setup_logging(config)
    
    # Initialize simulator
    simulator = FloodSimulator(config, logger)
    
    # Helper: Jamaica climate defaults (simple distributional defaults)
    def get_jamaica_defaults():
        # Typical Jamaica bounding box and rainfall tendencies
        defaults = {
            'bounds': (17.7, -78.4, 18.5, -76.1),  # south, west, north, east
            'dem_size': (200, 200),
            'elevation_range': (0, 1200),  # Jamaica has peaks ~700-2000m; keep generous range
            # Intense event example: 50-200 mm in a single event; pick a heavy event
            'rainfall_mm': 150.0,
            'duration_hours': 6.0
        }
        return defaults

    # Generate DEM over Jamaica-like area; prefer a real DEM if available
    print("Preparing DEM (real DEM if available, else synthetic)...")
    defaults = get_jamaica_defaults()
    dem = None
    bounds = defaults['bounds']
    dem_path = os.path.join('data', 'dem_jamaica.tif')
    
    # Try to fetch SRTM if requested and not already present
    if use_srtm and not os.path.exists(dem_path):
        print("Attempting to fetch SRTM data...")
        fetched = simulator.fetch_srtm(bounds, dem_path)
        if fetched:
            print(f"Successfully fetched SRTM to {fetched}")
    
    if RASTERIO_AVAILABLE and os.path.exists(dem_path):
        try:
            dem, bounds = simulator.load_dem_from_raster(dem_path, target_shape=defaults['dem_size'])
            print(f"Loaded DEM from {dem_path}")
        except Exception as e:
            logger.warning(f"Failed loading DEM from {dem_path}: {e}. Falling back to synthetic DEM.")

    if dem is None:
        dem = simulator.generate_synthetic_dem(size=defaults['dem_size'], elevation_range=defaults['elevation_range'])

    # Try to fetch CHIRPS if requested
    rainfall_field = None
    if use_chirps and chirps_start and chirps_end:
        print(f"Attempting to download CHIRPS for {chirps_start} to {chirps_end}...")
        rainfall_field = simulator.download_chirps_sum(chirps_start, chirps_end, bounds, target_shape=defaults['dem_size'])
        if rainfall_field is not None:
            print(f"Successfully downloaded CHIRPS (sum range: {rainfall_field.min():.1f}-{rainfall_field.max():.1f} mm)")

    # If no CHIRPS, generate a spatial rainfall field representative of Jamaica (orographic + convective)
    if rainfall_field is None:
        print(f"Generating synthetic spatial rainfall field totalling ~{defaults['rainfall_mm']} mm...")
        rainfall_field = simulator.generate_spatial_rainfall(dem, total_mm=defaults['rainfall_mm'], pattern='both', n_hotspots=5)

    # Simulate rainfall-runoff using the spatial rainfall field
    print("Simulating flood with spatial rainfall field...")
    water_depth = simulator.simulate_rainfall_runoff(dem, duration_hours=defaults['duration_hours'], rainfall_field=rainfall_field)

    # Visualize: static plot + interactive map (if folium installed)
    print("Creating static visualization (results/flood_simulation.png) and interactive map (results/flood_map.html)...")
    simulator.plot_simulation(dem, water_depth, save_path='results/flood_simulation.png')
    simulator.export_map(dem, water_depth, save_html='results/flood_map.html', bounds=defaults['bounds'])

    print("\nSimulation complete!")
    print(f"Maximum water depth: {water_depth.max():.3f} m")
    print(f"Average water depth: {water_depth.mean():.3f} m")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Flood simulation for Jamaica with optional real data sources.')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive plotting (GUI). Default: non-interactive (Agg backend).')
    parser.add_argument('--use-srtm', action='store_true', help='Attempt to fetch SRTM DEM if not already present at data/dem_jamaica.tif')
    parser.add_argument('--use-chirps', action='store_true', help='Attempt to download CHIRPS daily rainfall data.')
    parser.add_argument('--chirps-start', type=str, default='', help='CHIRPS download start date (YYYY-MM-DD). Required if --use-chirps.')
    parser.add_argument('--chirps-end', type=str, default='', help='CHIRPS download end date (YYYY-MM-DD). Required if --use-chirps.')
    
    args = parser.parse_args()
    
    # Switch to interactive backend if requested
    if args.interactive:
        matplotlib.use('TkAgg')
    
    main(interactive=args.interactive, use_srtm=args.use_srtm, use_chirps=args.use_chirps,
         chirps_start=args.chirps_start, chirps_end=args.chirps_end)
