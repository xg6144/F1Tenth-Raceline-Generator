import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from map_loader import MapLoader
from path_editor import PathModel

def verify():
    print("--- Verifying Logic with Enhancements ---")
    
    # 1. Map Loader (Large Map)
    try:
        # PGM 500x500
        loader = MapLoader("assets/test_map.yaml")
        print(f"[PASS] Map Loaded. Shape: {loader.occupancy_grid.shape}")
        if loader.occupancy_grid.shape != (500, 500):
            print(f"[FAIL] Map size mismatch. {loader.occupancy_grid.shape} != (500, 500)")
        
    except Exception as e:
        print(f"[FAIL] Map Load: {e}")

    # 2. Path Model with Velocity
    try:
        model = PathModel()
        # Straight line
        model.add_waypoint(0, 0)
        model.add_waypoint(10, 0)
        model.update_path(max_v=5.0)
        
        # Check velocity const
        v0 = model.interpolated_path[0][2]
        print(f"Straight Line Velocity: {v0}")
        assert abs(v0 - 5.0) < 0.1
        
        # Curve
        model = PathModel()
        model.add_waypoint(0, 0)
        model.add_waypoint(5, 5) # Turn
        model.add_waypoint(10, 0)
        model.update_path(max_v=10.0, max_lat_acc=1.0)
        
        # Middle point should be slower
        mid_idx = len(model.interpolated_path) // 2
        mid_v = model.interpolated_path[mid_idx][2]
        print(f"Curve Middle Velocity: {mid_v:.2f}")
        
        # Check if slower than max
        if mid_v < 10.0:
            print("[PASS] Velocity reduced on curve")
        else:
            print("[WARN] Velocity might not be reduced enough or curve too gentle")
            
    except Exception as e:
        print(f"[FAIL] Velocity Logic: {e}")

if __name__ == "__main__":
    verify()
