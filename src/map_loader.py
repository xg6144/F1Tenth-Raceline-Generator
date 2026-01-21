import yaml
import numpy as np
from PIL import Image
import os

class MapLoader:
    def __init__(self, map_yaml_path):
        self.map_yaml_path = map_yaml_path
        self.map_dir = os.path.dirname(map_yaml_path)
        self.metadata = self._load_yaml()
        self.image_path = os.path.join(self.map_dir, self.metadata['image'])
        self.resolution = self.metadata['resolution']
        self.origin = self.metadata['origin'] # [x, y, theta]
        self.occupancy_grid = self._load_pgm()

    def _load_yaml(self):
        with open(self.map_yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def _load_pgm(self):
        try:
            image = Image.open(self.image_path)
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array (row-major: [y, x])
            # PGM usually has (0,0) at top-left.
            # ROS maps: usually (0,0) is lower-left in world coords, 
            # but image loaded is top-left.
            # We will keep it as image coords for now and handle conversion later.
            return np.array(image)
        except Exception as e:
            print(f"Error loading PGM: {e}")
            return None

    def save_pgm(self, output_path=None):
        """
        Save the current occupancy grid to a PGM file.
        """
        if self.occupancy_grid is None:
            return False
        path = output_path if output_path else self.image_path
        try:
            img = Image.fromarray(self.occupancy_grid.astype(np.uint8))
            img.save(path)
            return True
        except Exception as e:
            print(f"Error saving PGM: {e}")
            return False

    def pixel_to_world(self, px, py):
        """
        Convert pixel coordinates (image frame) to world coordinates.
        Image frame: (0,0) top-left, x right, y down.
        World frame: (0,0) at origin specified in yaml, usually aligned with lower-left of map *if* map was not flipped.
        
        Standard ROS map_server:
        World (x,y) = Origin + (Pixel(x,y) * resolution)
        BUT, image (0,0) is top-left. World grid usually assumes (0,0) is bottom-left relative to data layout.
        
        Actually, let's stick to standard formula if we flip the image or not.
        Let's assume the image data[y, x] corresponds to world (x * res, (height-1-y) * res) + origin.
        """
        if self.occupancy_grid is None:
            return 0, 0
        
        height, width = self.occupancy_grid.shape
        
        # Invert Y for image-to-world conversion because image origin is top-left
        # world origin is typically bottom-left relative to the grid.
        # Adjusted Py = height - 1 - py
        
        world_x = self.origin[0] + (px * self.resolution)
        world_y = self.origin[1] + ((height - 1 - py) * self.resolution)
        
        return world_x, world_y

    def world_to_pixel(self, wx, wy):
        """
        Convert world coordinates to pixel coordinates.
        """
        if self.occupancy_grid is None:
            return 0, 0

        height, width = self.occupancy_grid.shape

        # wx = origin_x + px * res  =>  px = (wx - origin_x) / res
        px = int((wx - self.origin[0]) / self.resolution)
        
        # wy = origin_y + (height - 1 - py) * res
        # (wy - origin_y) / res = height - 1 - py
        # py = height - 1 - ((wy - origin_y) / res)
        py = int(height - 1 - ((wy - self.origin[1]) / self.resolution))

        return px, py
