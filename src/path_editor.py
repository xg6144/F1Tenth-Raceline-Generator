import numpy as np
from scipy.interpolate import CubicSpline
import csv

class PathModel:
    def __init__(self):
        self.waypoints = [] # List of [x, y]
        self.interpolated_path = [] # List of [x, y]
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

    def update_path(self, max_v=5.0, max_lat_acc=2.0):
        if len(self.waypoints) < 2:
            self.interpolated_path = []
            self.segment_speeds = []
            return

        points = np.array(self.waypoints)
        
        if len(points) == 2:
            # Linear interpolation
            val = np.linspace(0, 1, 100)
            x_interp = points[0][0] + (points[1][0] - points[0][0]) * val
            y_interp = points[0][1] + (points[1][1] - points[0][1]) * val
            # Constant velocity for straight line
            v_interp = np.full_like(x_interp, max_v)
            self.interpolated_path = np.column_stack((x_interp, y_interp, v_interp)).tolist()
            self.segment_speeds = [float(max_v)]
            return

        # Spline Interpolation
        diffs = np.diff(points, axis=0)
        dists = np.sqrt((diffs**2).sum(axis=1))
        cumulative_dist = np.concatenate(([0], np.cumsum(dists)))
        
        try:
            cs_x = CubicSpline(cumulative_dist, points[:, 0])
            cs_y = CubicSpline(cumulative_dist, points[:, 1])
            
            # Generate denser points
            # Ensure enough density for curvature calculation
            total_dist = cumulative_dist[-1]
            num_samples = int(total_dist * 20) # 20 points per meter
            sample_dists = np.linspace(0, total_dist, num_samples)
            
            xs = cs_x(sample_dists)
            ys = cs_y(sample_dists)
            
            # Calculate Derivatives for Curvature
            dx = cs_x(sample_dists, 1)
            dy = cs_y(sample_dists, 1)
            ddx = cs_x(sample_dists, 2)
            ddy = cs_y(sample_dists, 2)
            
            # Curvature: k = |dx * ddy - dy * ddx| / (dx^2 + dy^2)^(3/2)
            numerator = np.abs(dx * ddy - dy * ddx)
            denominator = (dx**2 + dy**2)**1.5
            curvature = numerator / (denominator + 1e-6) # Avoid division by zero
            
            # Velocity: v = min(max_v, sqrt(max_lat_acc / k))
            # k can be 0, so handle that.
            
            velocities = []
            for k in curvature:
                if k < 1e-4:
                    v = max_v
                else:
                    v = np.sqrt(max_lat_acc / k)
                velocities.append(min(max_v, v))

            velocities = np.array(velocities, dtype=float)
            
            self.interpolated_path = np.column_stack((xs, ys, velocities)).tolist()

            # Compute average speed for each original waypoint segment
            segment_speeds = []
            for i in range(len(points) - 1):
                start = cumulative_dist[i]
                end = cumulative_dist[i + 1]
                mask = (sample_dists >= start) & (sample_dists <= end)
                if np.any(mask):
                    seg_v = float(np.mean(velocities[mask]))
                else:
                    # Fallback to nearest sample to the segment midpoint
                    midpoint = 0.5 * (start + end)
                    idx = int(np.argmin(np.abs(sample_dists - midpoint)))
                    seg_v = float(velocities[idx])
                segment_speeds.append(seg_v)

            self.segment_speeds = segment_speeds
        except Exception as e:
            print(f"Spline/Velocity error: {e}")
            # Fallback
            self.interpolated_path = []
            self.segment_speeds = []
            for p in points:
                 self.interpolated_path.append([p[0], p[1], 1.0])

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
        Export path in a format compatible with the MPC node using user waypoints
        (not dense interpolation) to avoid oversampling.
        Columns (semicolon-separated, no header):
        0: index, 1: x, 2: y, 3: yaw, 4: placeholder (0.0), 5: speed
        """
        if len(self.waypoints) < 2:
            return False
        try:
            pts = np.array(self.waypoints, dtype=float)
            xs, ys = pts[:, 0], pts[:, 1]

            # Yaw from neighbor points
            yaws = []
            for i in range(len(pts)):
                prev_idx = max(0, i - 1)
                next_idx = min(len(pts) - 1, i + 1)
                dx = xs[next_idx] - xs[prev_idx]
                dy = ys[next_idx] - ys[prev_idx]
                yaws.append(float(np.arctan2(dy, dx)))

            # Speeds from segment_speeds (len-1). Use adjacent segment average.
            speeds = []
            for i in range(len(pts)):
                if not self.segment_speeds:
                    speeds.append(0.0)
                    continue
                if i == len(pts) - 1:
                    speeds.append(float(self.segment_speeds[-1]))
                else:
                    speeds.append(float(self.segment_speeds[i]))

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                for idx, (x, y, yaw, v) in enumerate(zip(xs, ys, yaws, speeds)):
                    writer.writerow([idx, x, y, yaw, 0.0, v])
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
