# gui.py

import sys
import numpy as np
import math
from collections import deque
from scipy import ndimage
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QAction, QFileDialog, 
                             QMessageBox, QLabel, QStatusBar, QDoubleSpinBox, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton, QMenu, QCheckBox, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QImage, QColor, QPixmap, QPainterPath, QFont

from map_loader import MapLoader
from path_editor import PathModel

class MapCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.map_loader = None
        self.path_model = PathModel()
        
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        
        self.last_mouse_pos = QPointF(0, 0)
        self.dragging_map = False
        self.dragging_waypoint_idx = -1
        self.space_move_active = False
        self.edit_mode = False
        self.brush_color = 255  # default white
        self.brush_size = 1
        self.cursor_pos = QPointF(0, 0)
        self.highlight_index = -1
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.editor_mode = 'ADD' # Only add; move is triggered with spacebar

    def set_map(self, map_loader):
        self.map_loader = map_loader
        self.path_model = PathModel() # Reset path when loading new map
        self.highlight_index = -1
        self.fit_map_to_view()
        if self.parent():
            mw = self.parent().window()
            if isinstance(mw, MainWindow):
                mw.update_table()
        self.update()

    def set_mode(self, mode):
        # Only ADD mode is available; kept for compatibility with existing UI wiring.
        self.editor_mode = 'ADD'

    def set_edit_mode(self, enabled):
        self.edit_mode = enabled
        if enabled:
            self.dragging_waypoint_idx = -1
            self.space_move_active = False
            self.highlight_index = -1

    def set_highlight(self, idx):
        self.highlight_index = idx
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw Map
        if self.map_loader and self.map_loader.occupancy_grid is not None:
            # Transform painter for map
            painter.save()
            painter.translate(self.offset)
            painter.scale(self.scale, self.scale)
            
            # Convert numpy array to QImage
            # PGM is grayscale 0-255. 0=Black(Occupied), 255=White(Free) usually.
            # But standard PGM might be generic.
            # We assume uint8.
            h, w = self.map_loader.occupancy_grid.shape
            img_data = self.map_loader.occupancy_grid.astype('uint8').tobytes()
            qimg = QImage(img_data, w, h, w, QImage.Format_Grayscale8)
            
            painter.drawImage(0, 0, qimg)
            
            # 2. Draw Path
            self._draw_path(painter)
            self._draw_waypoints(painter)
            
            painter.restore()
        else:
            painter.drawText(self.rect(), Qt.AlignCenter, "Load a Map (File -> Open YAML)")

        # Brush preview (drawn in screen coords)
        if self.edit_mode:
            painter.save()
            pen = QPen(QColor(50, 50, 50, 160), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            half = self.brush_size * self.scale
            x = int(round(self.cursor_pos.x() - half))
            y = int(round(self.cursor_pos.y() - half))
            size = int(round(half * 2))
            painter.drawRect(x, y, size, size)
            painter.restore()

    def _draw_path(self, painter):
        if not self.path_model.interpolated_path:
            return

        pen = QPen(QColor(0, 0, 255), 2)
        painter.setPen(pen)
        
        # Convert path points (world coords) to pixel coords
        # This is tricky because we need to use map_loader's transform
        # The painter is already in IMAGE PIXEL SPACE due to translate/scale above? 
        # Wait, translate/scale applies to the painter's coordinate system.
        # If we draw at (x,y), it will be transformed.
        # So we should draw in IMAGE PIXEL coordinates.
        
        path_pixels = []
        for point in self.path_model.interpolated_path:
            x, y = point[0], point[1]
            px, py = self.map_loader.world_to_pixel(x, y)
            path_pixels.append(QPointF(px, py))
            
        if len(path_pixels) > 1:
            qpath = QPainterPath()
            qpath.moveTo(path_pixels[0])
            for p in path_pixels[1:]:
                qpath.lineTo(p)
            painter.drawPath(qpath)

    def _draw_waypoints(self, painter):
        if self.edit_mode:
            return
        base_color = QColor(255, 0, 0)
        highlight_color = QColor(0, 180, 180)
        pen = QPen(base_color, 2)
        brush = QBrush(base_color)
        painter.setPen(pen)
        painter.setBrush(brush)
        label_font = QFont(painter.font())
        label_font.setPointSize(3)
        painter.setFont(label_font)
        
        for i, (x, y) in enumerate(self.path_model.waypoints):
            px, py = self.map_loader.world_to_pixel(x, y)
            radius = 1
            if i == self.highlight_index:
                painter.setPen(QPen(highlight_color, 2))
                painter.setBrush(QBrush(highlight_color))
            else:
                painter.setPen(QPen(base_color, 2))
                painter.setBrush(QBrush(base_color))
            painter.drawEllipse(QPointF(px, py), radius, radius)
            
            # Draw index with segment speed (if available)
            label = str(i)
            if i < len(self.path_model.segment_speeds):
                label = f"{label} ({self.path_model.segment_speeds[i]:.1f} m/s)"
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(QPointF(px + 5, py - 5), label)
            painter.setPen(pen) # Reset pen for next ellipse

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.1 if zoom_in else 0.9
        
        # Zoom centered on mouse
        old_pos = self.mapFromGlobal(event.globalPos())
        self.scale *= factor
        self.update()

    def update_path_params(self, max_v, max_lat_acc):
         self.path_model.update_path(max_v, max_lat_acc)
         self.update()

    def mousePressEvent(self, event):
        if not self.map_loader:
            # When no map is loaded, any click opens the map file dialog
            if self.parent():
                mw = self.parent().window()
                if isinstance(mw, MainWindow):
                    mw.load_map()
            return

        pos = event.pos()
        self.cursor_pos = pos
        # Convert widget pos to image pixels
        # widget_x = offset_x + image_x * scale
        # image_x = (widget_x - offset_x) / scale
        img_x = (pos.x() - self.offset.x()) / self.scale
        img_y = (pos.y() - self.offset.y()) / self.scale
        
        # Convert image pixels to world
        wx, wy = self.map_loader.pixel_to_world(img_x, img_y)

        if event.button() == Qt.MiddleButton:
            self.dragging_map = True
            self.last_mouse_pos = pos
        elif event.button() == Qt.LeftButton:
            if self.edit_mode:
                self._paint_at(img_x, img_y)
            elif self.space_move_active:
                idx = self.path_model.get_closest_waypoint_index(wx, wy, threshold=0.5)
                if idx != -1:
                    self.dragging_waypoint_idx = idx
            else:
                self.path_model.add_waypoint(wx, wy)
                # Re-param is expensive, maybe just add raw? 
                # But we need auto-calc. Let's signal MainWindow or just grab default?
                # Ideally MainWindow holds the params.
                # For now let's use default/last params or get from parent?
                # Better: trigger a signal. But simple approach:
                if self.parent():
                     mw = self.parent().window() # Get MainWindow
                     if isinstance(mw, MainWindow):
                         mw.recalc_path()
                     else:
                        self.update() # Fallback
                else:
                    self.update()
        elif event.button() == Qt.RightButton:
            if self.edit_mode:
                return
            idx = self.path_model.get_closest_waypoint_index(wx, wy, threshold=0.5)
            if idx != -1:
                self._show_insert_menu(event.globalPos(), idx, wx, wy)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        self.cursor_pos = pos
        
        if self.dragging_map:
            delta = pos - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = pos
            self.update()
        elif self.dragging_waypoint_idx != -1:
            img_x = (pos.x() - self.offset.x()) / self.scale
            img_y = (pos.y() - self.offset.y()) / self.scale
            wx, wy = self.map_loader.pixel_to_world(img_x, img_y)
            self.path_model.move_waypoint(self.dragging_waypoint_idx, wx, wy)
            self.update()
        elif self.edit_mode and event.buttons() & Qt.LeftButton:
            img_x = (pos.x() - self.offset.x()) / self.scale
            img_y = (pos.y() - self.offset.y()) / self.scale
            self._paint_at(img_x, img_y)
        else:
            if self.edit_mode:
                self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_map = False
        prev_drag_idx = self.dragging_waypoint_idx
        self.dragging_waypoint_idx = -1
        # After moving a waypoint, recompute with current params
        if prev_drag_idx != -1 and self.parent():
            mw = self.parent().window()
            if isinstance(mw, MainWindow):
                mw.recalc_path()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # Toggle move mode
            if not self.edit_mode:
                self.space_move_active = not self.space_move_active
                if self.parent():
                    mw = self.parent().window()
                    if isinstance(mw, MainWindow):
                        mw.status_bar.showMessage(f"Move mode {'ON' if self.space_move_active else 'OFF'}")
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # Close loop and refresh path/table
            if self.map_loader and self.path_model.waypoints:
                self.path_model.close_loop_if_needed()
                if self.parent():
                    mw = self.parent().window()
                    if isinstance(mw, MainWindow):
                        mw.recalc_path()
                        return
                self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)

    def _show_insert_menu(self, global_pos, idx, wx, wy):
        """
        Context menu to insert a waypoint before or after the clicked waypoint.
        """
        menu = QMenu(self)
        before_action = menu.addAction("Insert Before")
        after_action = menu.addAction("Insert After")
        chosen = menu.exec_(global_pos)
        if not chosen:
            return

        insert_idx = idx
        if chosen == after_action:
            insert_idx = idx + 1

        self.path_model.insert_waypoint(insert_idx, wx, wy)
        if self.parent():
            mw = self.parent().window()
            if isinstance(mw, MainWindow):
                mw.recalc_path()
                return
        self.update()

    def _paint_at(self, img_x, img_y):
        """
        Paint on the occupancy grid in image coordinates.
        """
        if not self.map_loader or self.map_loader.occupancy_grid is None:
            return
        h, w = self.map_loader.occupancy_grid.shape
        cx = int(round(img_x))
        cy = int(round(img_y))
        r = max(1, int(self.brush_size))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r)
        self.map_loader.occupancy_grid[y0:y1, x0:x1] = self.brush_color
        self.update()

    def fit_map_to_view(self):
        """
        Scale and center the map so it fills most of the canvas when loaded.
        """
        if not self.map_loader or self.map_loader.occupancy_grid is None:
            return
        if self.width() == 0 or self.height() == 0:
            return

        h, w = self.map_loader.occupancy_grid.shape
        if w == 0 or h == 0:
            return

        padding = 0.9
        scale_x = self.width() / w
        scale_y = self.height() / h
        self.scale = min(scale_x, scale_y) * padding

        map_w = w * self.scale
        map_h = h * self.scale
        self.offset = QPointF((self.width() - map_w) / 2, (self.height() - map_h) / 2)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("F1TENTH Path Generator")
        self.resize(1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)

        # Top area: canvas + waypoint table
        top_layout = QHBoxLayout()
        self.canvas = MapCanvas(self) # Pass self as parent used in mouse events lookup
        top_layout.addWidget(self.canvas, 3)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Index", "s (m)", "X", "Y", "Psi (rad)", "Kappa (rad/m)", "vx (m/s)", "ax (m/s^2)"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(240)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        top_layout.addWidget(self.table, 1)
        layout.addLayout(top_layout)
        
        # Settings Bar
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Max Speed (m/s):"))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 20.0)
        self.spin_speed.setValue(5.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.valueChanged.connect(self.recalc_path)
        settings_layout.addWidget(self.spin_speed)
        
        settings_layout.addWidget(QLabel("Max Lat Accel (m/s^2):"))
        self.spin_lat_acc = QDoubleSpinBox()
        self.spin_lat_acc.setRange(0.1, 20.0)
        self.spin_lat_acc.setValue(2.0)
        self.spin_lat_acc.setSingleStep(0.1)
        self.spin_lat_acc.valueChanged.connect(self.recalc_path)
        settings_layout.addWidget(self.spin_lat_acc)

        auto_btn = QPushButton("Auto Raceline")
        auto_btn.clicked.connect(self.auto_generate_raceline)
        settings_layout.addWidget(auto_btn)

        # Map edit tools
        self.chk_edit_map = QCheckBox("Edit Map")
        self.chk_edit_map.stateChanged.connect(self.toggle_edit_map)
        settings_layout.addWidget(self.chk_edit_map)

        settings_layout.addWidget(QLabel("Color:"))
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["White (free)", "Gray (unknown)", "Black (wall)"])
        self.cmb_color.currentIndexChanged.connect(self.change_brush_color)
        settings_layout.addWidget(self.cmb_color)

        settings_layout.addWidget(QLabel("Brush:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 50)
        self.spin_brush.setValue(3)
        self.spin_brush.valueChanged.connect(self.change_brush_size)
        settings_layout.addWidget(self.spin_brush)

        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        self._create_actions()
        self._create_menubar()
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a map to begin.")
        self.canvas.setFocus()

    def _create_actions(self):
        self.load_action = QAction("Load Map...", self)
        self.load_action.triggered.connect(self.load_map)

        self.load_path_action = QAction("Load Path CSV...", self)
        self.load_path_action.triggered.connect(self.load_path)

        self.save_action = QAction("Save Path...", self)
        self.save_action.triggered.connect(self.save_path)

        self.save_map_action = QAction("Save Map...", self)
        self.save_map_action.triggered.connect(self.save_map)
        
        self.mode_add_action = QAction("Add Mode", self)
        self.mode_add_action.setCheckable(True)
        self.mode_add_action.setChecked(True)
        self.mode_add_action.triggered.connect(lambda: self.set_mode('ADD'))

    def _create_menubar(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.load_path_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_map_action)

    def set_mode(self, mode):
        self.canvas.set_mode(mode)
        self.mode_add_action.setChecked(True)
        self.status_bar.showMessage(f"Mode: {mode}")

    def recalc_path(self):
        max_v = self.spin_speed.value()
        acc = self.spin_lat_acc.value()
        self.canvas.update_path_params(max_v, acc)
        self.update_table()

    def auto_generate_raceline(self):
        """
        Generate a raceline by skeletonizing the free space (white) and tracing the largest loop.
        This approximates a medial-axis raceline similar to the global trajectory optimizer.
        """
        ml = self.canvas.map_loader
        if ml is None or ml.occupancy_grid is None:
            QMessageBox.information(self, "No Map", "Please load a map before generating a raceline.")
            return

        h, w = ml.occupancy_grid.shape
        if w < 2 or h < 2:
            QMessageBox.warning(self, "Map Error", "Map size is too small to generate a raceline.")
            return

        grid = ml.occupancy_grid.astype(np.uint8)
        free_mask = grid >= 250  # white = drivable
        if not np.any(free_mask):
            QMessageBox.warning(self, "Map Error", "No free space detected for raceline generation.")
            return

        skeleton = self._skeletonize_zhang_suen(free_mask)
        coords = np.argwhere(skeleton)  # (row, col)
        if coords.size == 0:
            QMessageBox.warning(self, "Map Error", "Skeletonization failed to find a path.")
            return

        largest = self._largest_component(coords)
        if len(largest) < 4:
            QMessageBox.warning(self, "Map Error", "No sizeable loop found in skeleton.")
            return

        ordered = self._extract_cycle(largest)
        if len(ordered) < 4:
            ordered = self._trace_ordered_loop(largest)
        if len(ordered) < 4:
            QMessageBox.warning(self, "Map Error", "Ordering the skeleton loop failed.")
            return

        # Downsample to limit number of points (approx 200)
        step = max(1, len(ordered) // 200)
        ordered = ordered[::step]
        if len(ordered) < 4:
            QMessageBox.warning(self, "Map Error", "Too few points after downsampling.")
            return

        # Smooth with moving average to reduce jagged segments
        if len(ordered) >= 5:
            kernel = 5
            padded = np.vstack([ordered[-2:], ordered, ordered[:2]])
            smooth = []
            for i in range(2, len(padded) - 2):
                window = padded[i-2:i+3]
                smooth.append(np.mean(window, axis=0))
            ordered = np.array(smooth, dtype=float)

        # Optimize centerline to maximize clearance and smoothness (distance field + Laplacian)
        dist_map = ndimage.distance_transform_edt(free_mask)
        ordered = self._optimize_centerline(dist_map, ordered, iterations=60, alpha=0.25, beta=0.2)

        loop = []
        for py, px in ordered:  # coords are (y, x)
            wx, wy = ml.pixel_to_world(px, py)
            loop.append([wx, wy])
        loop.append(loop[0])  # close loop

        self.canvas.path_model.waypoints = loop
        self.recalc_path()
        self.status_bar.showMessage("Auto raceline generated from skeleton.")

    def toggle_edit_map(self, state):
        enabled = state == Qt.Checked
        self.canvas.set_edit_mode(enabled)
        if enabled:
            self.status_bar.showMessage("Map edit mode ON")
            self.canvas.update()
        else:
            self.status_bar.showMessage("Map edit mode OFF")
            self.canvas.update()

    def change_brush_color(self, idx):
        if idx == 0:
            val = 255  # white free
        elif idx == 1:
            val = 205  # gray unknown
        else:
            val = 0    # black wall
        self.canvas.brush_color = val
        self.status_bar.showMessage(f"Brush color set to {val}")

    def change_brush_size(self, val):
        self.canvas.brush_size = val

    def _skeletonize_zhang_suen(self, mask_bool):
        """
        Simple Zhang-Suen thinning to get a 1-pixel skeleton from free-space mask.
        mask_bool: bool array where True is drivable.
        """
        img = mask_bool.astype(np.uint8)
        changed = True
        while changed:
            changed = False
            img, step_changed = self._zs_iteration(img, step=0)
            changed = changed or step_changed
            img, step_changed = self._zs_iteration(img, step=1)
            changed = changed or step_changed
        return img.astype(bool)

    def _zs_iteration(self, img, step):
        """One Zhang-Suen sub-iteration."""
        padded = np.pad(img, 1, mode='constant')
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + \
                      ((p3 == 0) & (p4 == 1)).astype(np.uint8) + \
                      ((p4 == 0) & (p5 == 1)).astype(np.uint8) + \
                      ((p5 == 0) & (p6 == 1)).astype(np.uint8) + \
                      ((p6 == 0) & (p7 == 1)).astype(np.uint8) + \
                      ((p7 == 0) & (p8 == 1)).astype(np.uint8) + \
                      ((p8 == 0) & (p9 == 1)).astype(np.uint8) + \
                      ((p9 == 0) & (p2 == 1)).astype(np.uint8)

        cond1 = (img == 1)
        cond2 = (neighbors >= 2) & (neighbors <= 6)
        cond3 = (transitions == 1)
        if step == 0:
            cond4 = (p2 * p4 * p6 == 0)
            cond5 = (p4 * p6 * p8 == 0)
        else:
            cond4 = (p2 * p4 * p8 == 0)
            cond5 = (p2 * p6 * p8 == 0)

        remove = cond1 & cond2 & cond3 & cond4 & cond5
        if np.any(remove):
            img = img.copy()
            img[remove] = 0
            return img, True
        return img, False

    def _largest_component(self, coords):
        """
        Given coords (N x 2) of skeleton pixels (y, x), return coords of largest component.
        """
        if len(coords) == 0:
            return []
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[coords[:, 0], coords[:, 1]] = 1

        visited = np.zeros_like(mask, dtype=bool)
        best = []
        for y, x in coords:
            if visited[y, x]:
                continue
            comp = []
            dq = deque()
            dq.append((y, x))
            visited[y, x] = True
            while dq:
                cy, cx = dq.popleft()
                comp.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            dq.append((ny, nx))
            if len(comp) > len(best):
                best = comp
        return np.array(best, dtype=int)

    def _order_loop(self, coords):
        """
        Order loop points by angle around centroid to create a closed loop.
        coords: Nx2 array (y, x)
        """
        if len(coords) == 0:
            return []
        center = coords.mean(axis=0)
        angles = np.arctan2(coords[:, 0] - center[0], coords[:, 1] - center[1])
        order = np.argsort(angles)
        return coords[order]

    def _trace_ordered_loop(self, coords):
        """
        Trace skeleton as a loop using neighbor following after pruning leaves.
        coords: Nx2 array (y, x)
        """
        if len(coords) == 0:
            return []
        nodes = {tuple(c) for c in coords.tolist()}
        graph = {}
        for y, x in nodes:
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in nodes:
                        neigh.append((ny, nx))
            graph[(y, x)] = neigh

        # Prune leaves iteratively to keep closed loops
        deg = {n: len(neigh) for n, neigh in graph.items()}
        queue = deque([n for n, d in deg.items() if d <= 1])
        removed = set()
        while queue:
            n = queue.popleft()
            if n in removed:
                continue
            removed.add(n)
            for m in graph.get(n, []):
                if m in removed:
                    continue
                deg[m] -= 1
                if deg[m] == 1:
                    queue.append(m)
        # Build pruned graph
        pruned_nodes = [n for n in nodes if n not in removed]
        if len(pruned_nodes) < 4:
            return np.array([], dtype=float)
        graph_pruned = {}
        for y, x in pruned_nodes:
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in nodes and (ny, nx) not in removed:
                        neigh.append((ny, nx))
            graph_pruned[(y, x)] = neigh

        nodes = set(pruned_nodes)

        # Pick a start with degree 2 if possible
        start = None
        for n, neigh in graph_pruned.items():
            if len(neigh) == 2:
                start = n
                break
        if start is None:
            start = next(iter(graph_pruned.keys()))

        path = []
        visited_edges = set()
        current = start
        prev = None
        max_steps = len(nodes) * 6
        steps = 0
        while steps < max_steps:
            path.append(current)
            steps += 1
            neighs = graph_pruned.get(current, [])
            candidates = [n for n in neighs if n != prev and (current, n) not in visited_edges and (n, current) not in visited_edges]
            if not candidates:
                candidates = [n for n in neighs if n != prev]
            if not candidates:
                break
            if prev is not None:
                pv = np.array(current) - np.array(prev)
                best = None
                best_dot = -1e9
                for c in candidates:
                    cv = np.array(c) - np.array(current)
                    dot = np.dot(pv, cv)
                    if dot > best_dot:
                        best_dot = dot
                        best = c
                nxt = best
            else:
                nxt = candidates[0]
            visited_edges.add((current, nxt))
            prev, current = current, nxt
            if current == start and len(path) > 5:
                break
        return np.array(path, dtype=float)

    def _extract_cycle(self, coords):
        """
        Keep only nodes with degree==2 (simple cycles) by pruning others iteratively,
        then return the largest remaining cycle traced.
        """
        nodes = {tuple(c) for c in coords.tolist()}
        graph = {}
        for y, x in nodes:
            neigh = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in nodes:
                        neigh.append((ny, nx))
            graph[(y, x)] = neigh

        # prune nodes with degree != 2
        removed = set()
        changed = True
        while changed:
            changed = False
            to_remove = [n for n, neigh in graph.items() if len(neigh) != 2]
            if not to_remove:
                break
            for n in to_remove:
                if n in removed:
                    continue
                removed.add(n)
                changed = True
                for m in graph.get(n, []):
                    if m in removed:
                        continue
                    if n in graph.get(m, []):
                        graph[m] = [k for k in graph[m] if k != n]
                graph[n] = []

        cycles_nodes = [n for n in graph.keys() if n not in removed and len(graph[n]) == 2]
        if len(cycles_nodes) < 4:
            return np.array([], dtype=float)

        visited = set()
        best_path = []
        for start in cycles_nodes:
            if start in visited:
                continue
            path = []
            prev = None
            curr = start
            max_steps = len(cycles_nodes) + 5
            steps = 0
            while steps < max_steps:
                path.append(curr)
                visited.add(curr)
                neighs = [n for n in graph[curr] if n != prev]
                if not neighs:
                    break
                nxt = neighs[0]
                prev, curr = curr, nxt
                steps += 1
                if curr == start:
                    break
            if len(path) > len(best_path):
                best_path = path

        return np.array(best_path, dtype=float)

    def _optimize_centerline(self, dist_map, pts, iterations=40, alpha=0.2, beta=0.15):
        """
        Adjust centerline points to maximize clearance (follow distance gradient)
        and smoothness (Laplacian). Works in pixel space.
        """
        if len(pts) < 4:
            return pts
        pts = pts.astype(float)
        h, w = dist_map.shape
        gy, gx = np.gradient(dist_map)

        def clamp(p):
            p[0] = np.clip(p[0], 0, h - 1)
            p[1] = np.clip(p[1], 0, w - 1)
            return p

        for _ in range(iterations):
            new_pts = pts.copy()
            for i in range(len(pts)):
                prev = pts[i - 1]
                nex = pts[(i + 1) % len(pts)]
                cur = pts[i]
                lap = (prev + nex) / 2.0 - cur
                cy, cx = int(round(cur[0])), int(round(cur[1]))
                cy = np.clip(cy, 0, h - 1)
                cx = np.clip(cx, 0, w - 1)
                grad = np.array([gy[cy, cx], gx[cy, cx]])
                step_vec = alpha * grad + beta * lap
                candidate = cur + step_vec
                candidate = clamp(candidate)
                if dist_map[int(candidate[0]), int(candidate[1])] > 0:
                    new_pts[i] = candidate
                else:
                    new_pts[i] = cur
            pts = new_pts
        return pts

    def update_table(self):
        """
        Refresh the table with MPPI path fields.
        """
        wps = self.canvas.path_model.waypoints
        # Use simple sparse data we computed for waypoints
        waypoint_data = self.canvas.path_model.waypoint_data
        
        self.table.clearContents()
        self.table.setRowCount(len(wps))
        
        if len(wps) == 0:
            return

        for i, (x, y) in enumerate(wps):
            s_val = 0.0
            psi_val = 0.0
            k_val = 0.0
            v_val = 0.0
            ax_val = 0.0
            
            # Direct mapping by index if lengths match
            if waypoint_data and i < len(waypoint_data):
                pd = waypoint_data[i]
                s_val = pd['s_m']
                psi_val = pd['psi_rad']
                k_val = pd['kappa_radpm']
                v_val = pd['vx_mps']
                ax_val = pd['ax_mps2']

            self.table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{s_val:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{x:.3f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{y:.3f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{psi_val:.3f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{k_val:.3f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{v_val:.2f}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{ax_val:.2f}"))

        self.table.resizeColumnsToContents()
        if self.canvas.highlight_index >= len(wps):
            self.canvas.set_highlight(-1)

    def load_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Map YAML", "", "YAML Files (*.yaml);;All Files (*)")
        if path:
            try:
                loader = MapLoader(path)
                self.canvas.set_map(loader)
                self.status_bar.showMessage(f"Loaded map: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load map: {e}")

    def load_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Path CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        if not self.canvas.map_loader:
            QMessageBox.information(self, "No Map", "Load a map first to view the path.")
            return

        success = self.canvas.path_model.import_csv(path)
        if success:
            self.recalc_path()
            self.status_bar.showMessage(f"Loaded path: {path}")
        else:
            QMessageBox.warning(self, "Warning", "Failed to load path or path is invalid.")

    def save_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Path CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            success = self.canvas.path_model.export_csv(path)
            if success:
                self.status_bar.showMessage(f"Saved path to {path}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save path or path empty.")

    def save_map(self):
        if not self.canvas.map_loader or self.canvas.map_loader.occupancy_grid is None:
            QMessageBox.information(self, "No Map", "Load a map before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Map PGM", "", "PGM Files (*.pgm);;All Files (*)")
        if not path:
            return
        success = self.canvas.map_loader.save_pgm(path)
        if success:
            self.status_bar.showMessage(f"Saved map to {path}")
        else:
            QMessageBox.warning(self, "Warning", "Failed to save map.")

    def on_table_select(self):
        items = self.table.selectedItems()
        if not items:
            self.canvas.set_highlight(-1)
            return
        row = items[0].row()
        self.canvas.set_highlight(row)
