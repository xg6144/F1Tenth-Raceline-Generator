
import sys
import os
import math
import csv
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QMessageBox, QGraphicsView, 
                             QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, 
                             QHBoxLayout, QLabel, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QPointF

class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self._is_panning = False
        self.parent_window = parent

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0:
            self.scale(zoomInFactor, zoomInFactor)
        else:
            self.scale(zoomOutFactor, zoomOutFactor)

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.ShiftModifier:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._is_panning = True
            super().mousePressEvent(event)
        
        elif self.parent_window and self.parent_window.waypoint_mode_cb.isChecked() and event.button() == Qt.LeftButton:
            # Handle Waypoint Click
            scene_pos = self.mapToScene(event.pos())
            self.parent_window.add_waypoint(scene_pos)
        
        elif self.parent_window and self.parent_window.waypoint_mode_cb.isChecked() and event.button() == Qt.RightButton:
             # Remove last waypoint on right click
             self.parent_window.remove_last_waypoint()

        else:
            self.setDragMode(QGraphicsView.NoDrag)
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._is_panning:
            self.setDragMode(QGraphicsView.NoDrag)
            self._is_panning = False

class PGMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PGM Viewer & MPC Path Generator")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Controls Layout
        self.controls_layout = QHBoxLayout()
        self.layout.addLayout(self.controls_layout)

        self.open_button = QPushButton("Open PGM File")
        self.open_button.clicked.connect(self.open_file)
        self.controls_layout.addWidget(self.open_button)

        self.waypoint_mode_cb = QCheckBox("Edit Waypoints Mode")
        self.controls_layout.addWidget(self.waypoint_mode_cb)

        self.save_button = QPushButton("Save Path")
        self.save_button.clicked.connect(self.save_path)
        self.controls_layout.addWidget(self.save_button)
        
        self.info_label = QLabel("No Map Loaded")
        self.controls_layout.addWidget(self.info_label)

        # Graphics View
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self)
        self.view.setScene(self.scene)
        self.layout.addWidget(self.view)

        # Application State
        self.map_metadata = {"resolution": 0.05, "origin": [0.0, 0.0, 0.0]} # Default
        self.map_height = 0
        self.waypoints = [] # List of (scene_x, scene_y)
        self.waypoint_items = [] # List of QGraphicsEllipseItem
        self.path_lines = [] # List of QGraphicsLineItem

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PGM File", "", "PGM Files (*.pgm);;All Files (*)", options=options)
        
        if file_path:
            self.load_image(file_path)
            self.load_metadata(file_path)

    def load_image(self, file_path):
        try:
            image = QImage(file_path)
            if image.isNull():
                 QMessageBox.critical(self, "Error", "Failed to load image.")
                 return

            self.map_height = image.height()
            pixmap = QPixmap.fromImage(image)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(self.scene.itemsBoundingRect())
            
            # Reset Waypoints
            self.waypoints = []
            self.waypoint_items = []
            self.path_lines = []

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def load_metadata(self, pgm_path):
        yaml_path = pgm_path.replace(".pgm", ".yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.map_metadata["resolution"] = data.get("resolution", 0.05)
                    self.map_metadata["origin"] = data.get("origin", [0.0, 0.0, 0.0])
                    
                    res = self.map_metadata["resolution"]
                    origin = self.map_metadata["origin"]
                    self.info_label.setText(f"Res: {res} m/px | Origin: ({origin[0]:.2f}, {origin[1]:.2f})")
            except Exception as e:
                self.info_label.setText(f"Error loading YAML: {str(e)}")
        else:
            self.info_label.setText("No YAML found. Using defaults.")
            self.map_metadata = {"resolution": 0.05, "origin": [0.0, 0.0, 0.0]}

    def add_waypoint(self, scene_pos):
        x = scene_pos.x()
        y = scene_pos.y()
        self.waypoints.append((x, y))

        # Visuals
        radius = 5
        ellipse = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        ellipse.setBrush(QBrush(Qt.red))
        ellipse.setPen(QPen(Qt.black))
        self.scene.addItem(ellipse)
        self.waypoint_items.append(ellipse)

        # Draw Line
        if len(self.waypoints) > 1:
            last_x, last_y = self.waypoints[-2]
            line = QGraphicsLineItem(last_x, last_y, x, y)
            line.setPen(QPen(Qt.blue, 2))
            self.scene.addItem(line)
            self.path_lines.append(line)

    def remove_last_waypoint(self):
        if not self.waypoints:
            return
        
        self.waypoints.pop()
        
        # Remove Visuals
        if self.waypoint_items:
            item = self.waypoint_items.pop()
            self.scene.removeItem(item)
        
        if self.path_lines:
            line = self.path_lines.pop()
            self.scene.removeItem(line)

    def save_path(self):
        if not self.waypoints:
             QMessageBox.warning(self, "Warning", "No waypoints to save.")
             return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Path", "waypoints.csv", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if file_path:
            try:
                res = self.map_metadata["resolution"]
                origin_x = self.map_metadata["origin"][0]
                origin_y = self.map_metadata["origin"][1]
                
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["x_m", "y_m", "speed_mps"]) # Header
                    
                    for (u, v) in self.waypoints:
                        # Transform: scene(u,v) -> world(x,y)
                        # ROS Map Server: origin is bottom-left pixel.
                        # Image is top-down.
                        # so world_y = (image_height - v) * res + origin_y
                        
                        world_x = u * res + origin_x
                        world_y = (self.map_height - v) * res + origin_y
                        
                        writer.writerow([f"{world_x:.4f}", f"{world_y:.4f}", "1.0"]) # Default speed 1.0
                
                QMessageBox.information(self, "Success", f"Path saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save path: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PGMViewer()
    viewer.show()
    sys.exit(app.exec_())
