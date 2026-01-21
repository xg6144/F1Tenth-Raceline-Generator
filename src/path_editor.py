# path_editor.py

import numpy as np
from scipy.interpolate import CubicSpline
import csv

class PathModel:
    def __init__(self):
        self.waypoints = [] # List of [x, y]
        self.interpolated_path = [] # List of [x, y, v] for drawing
        self.path_data = [] # List of dicts with full MPPI fields (dense)
        self.waypoint_data = [] # List of dicts with full MPPI fields (sparse, at waypoints)
        self.velocity = 1.0 # Default velocity placeholder
        self.segment_speeds = [] # Average speed per segment (waypoint i -> i+1)

    def add_waypoint(self, x, y):
        self.waypoints.append([x, y])
        self.update_path()

    def remove_waypoint(self, index):
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)
            self.update_path()

    def insert_waypoint(self, index, x, y):
        """
        Insert waypoint at given index and re-run path update.
        """
        if index < 0:
            index = 0
        if index > len(self.waypoints):
            index = len(self.waypoints)
        self.waypoints.insert(index, [x, y])
        self.update_path()

    def move_waypoint(self, index, x, y):
        if 0 <= index < len(self.waypoints):
            self.waypoints[index] = [x, y]
            self.update_path()

    def update_path(self, max_v=5.0, max_lat_acc=2.0, max_long_acc=3.0):
        if len(self.waypoints) < 2:
            self.interpolated_path = []
            self.path_data = []
            self.segment_speeds = []
            return

        points = np.array(self.waypoints)
        
        # Check for closed loop
        is_closed = False
        if len(points) > 2:
            if np.linalg.norm(points[0] - points[-1]) < 0.1:
                is_closed = True
                points[-1] = points[0]

        # Parameterize by distance
        diffs = np.diff(points, axis=0)
        dists = np.sqrt((diffs**2).sum(axis=1))
        cumulative_dist = np.concatenate(([0], np.cumsum(dists)))
        total_dist = cumulative_dist[-1]
        
        if total_dist < 1e-3:
             self.interpolated_path = []
             self.path_data = []
             self.segment_speeds = []
             return

        # Sampling
        ds = 0.1 # 10cm resolution
        num_samples = int(total_dist / ds)
        if num_samples < 2:
            num_samples = 2
        
        s_vals = np.linspace(0, total_dist, num_samples)
        
        try:
            # Spline Interpolation
            cs_x = CubicSpline(cumulative_dist, points[:, 0], bc_type='not-a-knot')
            cs_y = CubicSpline(cumulative_dist, points[:, 1], bc_type='not-a-knot')
            
            xs = cs_x(s_vals)
            ys = cs_y(s_vals)
            
            # Derivatives
            dx = cs_x(s_vals, 1)
            dy = cs_y(s_vals, 1)
            ddx = cs_x(s_vals, 2)
            ddy = cs_y(s_vals, 2)
            
            # Heading (Psi)
            psis = np.arctan2(dy, dx)
            # Unwrap psi?
            psis = np.unwrap(psis)
            
            # Curvature (Kappa)
            # k = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
            numerator = dx * ddy - dy * ddx
            denominator = (dx**2 + dy**2)**1.5
            kappas = numerator / (denominator + 1e-6)
            
            # --- Velocity Profiling (Forward-Backward) ---
            
            # 1. Max velocity based on curvature (V_max_k)
            # v^2 <= a_lat_max / |k|
            # Add epsilon to k
            v_max_k = np.sqrt(max_lat_acc / (np.abs(kappas) + 1e-6))
            v_ref = np.minimum(max_v, v_max_k)
            
            # 2. Forward Pass
            # v_i+1^2 = v_i^2 + 2 * a * ds
            # We are limited by max_long_acc (acceleration)
            # v[i+1] <= sqrt(v[i]^2 + 2 * a_acc * ds)
            v_fwd = np.zeros_like(v_ref)
            v_fwd[0] = v_ref[0] # Start condition? Or 0? Assume valid entry speed.
            
            for i in range(len(v_ref) - 1):
                # Distance to next point
                ds_i = s_vals[i+1] - s_vals[i]
                v_next_sq = v_fwd[i]**2 + 2 * max_long_acc * ds_i
                v_fwd[i+1] = min(v_ref[i+1], np.sqrt(v_next_sq))
                
            # 3. Backward Pass
            # v[i] <= sqrt(v[i+1]^2 + 2 * a_dec * ds)
            # a_dec is absolute value here for formula
            v_bwd = np.zeros_like(v_ref)
            v_bwd[-1] = v_ref[-1] # End condition
            
            for i in range(len(v_ref) - 2, -1, -1):
                ds_i = s_vals[i+1] - s_vals[i]
                v_curr_sq = v_bwd[i+1]**2 + 2 * max_long_acc * ds_i
                v_bwd[i] = min(v_ref[i], np.sqrt(v_curr_sq))
                
            # Final velocity
            vs = np.minimum(v_fwd, v_bwd)
            
            # --- Acceleration Calculation ---
            # ax = (v_next^2 - v_curr^2) / (2*ds) approx
            axs = np.zeros_like(vs)
            # Central difference or forward?
            # Forward for i, Backward for end.
            for i in range(len(vs) - 1):
                ds_i = s_vals[i+1] - s_vals[i]
                axs[i] = (vs[i+1]**2 - vs[i]**2) / (2 * ds_i + 1e-6)
            axs[-1] = axs[-2] # Repeat last
            
            # --- Pack Data ---
            self.interpolated_path = []
            self.path_data = []
            
            for i in range(len(s_vals)):
                # Normalizing psi to -pi, pi for standard output if needed, but unwrap is good for diffs
                # User request: "-pi to +pi rad"
                p_norm = (psis[i] + np.pi) % (2 * np.pi) - np.pi
                
                row = {
                    's_m': float(s_vals[i]),
                    'x_m': float(xs[i]),
                    'y_m': float(ys[i]),
                    'psi_rad': float(p_norm),
                    'kappa_radpm': float(kappas[i]),
                    'vx_mps': float(vs[i]),
                    'ax_mps2': float(axs[i])
                }
                self.path_data.append(row)
                self.interpolated_path.append([xs[i], ys[i], vs[i]])
                
            # Update segment speeds for UI table (approximate)
            self.segment_speeds = []
            for i in range(len(points) - 1):
                s_start = cumulative_dist[i]
                s_end = cumulative_dist[i+1]
                # Find samples in this range
                mask = (s_vals >= s_start) & (s_vals <= s_end)
                if np.any(mask):
                    avg_v = np.mean(vs[mask])
                else:
                    avg_v = max_v # fallback
                self.segment_speeds.append(float(avg_v))
            
            # --- Map back to Waypoints ---
            # Interpolate vs, axs, psis, kappas at cumulative_dist
            # Note: psis should be unwrapped before interp, then wrapped back if needed
            # For accurate curvature at waypoints, using spline derivatives at s is better than interp
            
            wp_xs = cs_x(cumulative_dist)
            wp_ys = cs_y(cumulative_dist)
            wp_dx = cs_x(cumulative_dist, 1)
            wp_dy = cs_y(cumulative_dist, 1)
            wp_ddx = cs_x(cumulative_dist, 2)
            wp_ddy = cs_y(cumulative_dist, 2)
            
            wp_psis = np.arctan2(wp_dy, wp_dx)
            
            wp_num = wp_dx * wp_ddy - wp_dy * wp_ddx
            wp_den = (wp_dx**2 + wp_dy**2)**1.5
            wp_kappas = wp_num / (wp_den + 1e-6)
            
            # Velocity and Accel: Interpolate from dense profile
            wp_vs = np.interp(cumulative_dist, s_vals, vs)
            wp_axs = np.interp(cumulative_dist, s_vals, axs)
            
            self.waypoint_data = []
            for i in range(len(cumulative_dist)):
                 p_norm = (wp_psis[i] + np.pi) % (2 * np.pi) - np.pi
                 row = {
                    's_m': float(cumulative_dist[i]),
                    'x_m': float(wp_xs[i]),
                    'y_m': float(wp_ys[i]),
                    'psi_rad': float(p_norm),
                    'kappa_radpm': float(wp_kappas[i]),
                    'vx_mps': float(wp_vs[i]),
                    'ax_mps2': float(wp_axs[i])
                }
                 self.waypoint_data.append(row)
                
        except Exception as e:
            print(f"Update Path Error: {e}")
            self.interpolated_path = []
            self.path_data = []
            self.waypoint_data = []
            self.segment_speeds = []

    def close_loop_if_needed(self, tol=1e-3):
        """
        If the path is not closed, append the first waypoint to the end to close the loop.
        """
        if len(self.waypoints) < 3:
            return
        first = np.array(self.waypoints[0])
        last = np.array(self.waypoints[-1])
        if np.linalg.norm(first - last) > tol:
            self.waypoints.append(self.waypoints[0][:])

    def export_csv(self, filepath):
        """
        Export generated path with fields:
        s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
        """
        if not self.path_data:
            return False
            
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])
                # Export only user waypoints (requested by user)
                # Use self.waypoint_data instead of self.path_data
                data_curr = self.waypoint_data if self.waypoint_data else self.path_data
                
                for row in data_curr:
                    writer.writerow([
                        f"{row['s_m']:.4f}",
                        f"{row['x_m']:.4f}",
                        f"{row['y_m']:.4f}",
                        f"{row['psi_rad']:.4f}",
                        f"{row['kappa_radpm']:.4f}",
                        f"{row['vx_mps']:.4f}",
                        f"{row['ax_mps2']:.4f}"
                    ])
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    def import_csv(self, filepath):
        """
        Import path from CSV. Supports ';' or ',' delimiter.
        Expected columns: index, x, y, yaw, *, speed (at least x, y).
        """
        try:
            with open(filepath, 'r', newline='') as f:
                sample = f.read(1024)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=';,')
                    delim = dialect.delimiter
                except Exception:
                    delim = ';'

                reader = csv.reader(f, delimiter=delim)
                rows = list(reader)

            # Filter out empty rows
            rows = [r for r in rows if r]
            if not rows:
                return False

            waypoints = []
            for r in rows:
                # tolerate header-like first row if non-numeric
                try:
                    float(r[1])
                except Exception:
                    # assume header row, skip
                    continue
                if len(r) < 3:
                    continue
                try:
                    x = float(r[1])
                    y = float(r[2])
                    waypoints.append([x, y])
                except Exception:
                    continue

            if len(waypoints) < 2:
                return False

            self.waypoints = waypoints
            self.update_path()
            return True
        except Exception as e:
            print(f"Import error: {e}")
            return False

    def get_closest_waypoint_index(self, x, y, threshold=0.5):
        """
        Finds the index of the waypoint closest to (x, y) within threshold.
        """
        if not self.waypoints:
            return -1
        
        points = np.array(self.waypoints)
        dists = np.sqrt(np.sum((points - [x, y])**2, axis=1))
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < threshold:
            return min_idx
        return -1
