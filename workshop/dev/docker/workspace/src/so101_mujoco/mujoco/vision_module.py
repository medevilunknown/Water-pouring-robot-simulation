import mujoco
import numpy as np
import cv2
from ultralytics import YOLO
import math

class VisionModule:
    """
    Perception system for SO-101 robot.
    Integrates YOLOv8 for 2D detection and Depth-based projection for 3D world coordinates.
    """
    def __init__(self, m, d, model_path="yolov8n.pt"):
        self.m = m
        self.d = d
        print(f"Loading YOLO model: {model_path}")
        self.yolo = YOLO(model_path)
        self.renderer = mujoco.Renderer(m, 480, 640)
        self.renderer.enable_depth_rendering()
        
    def detect_3d(self, cam_name="external_camera"):
        """
        Detects objects (bottle, cup) and returns their 3D world coordinates.
        Returns: { 'bottle': [x,y,z], 'cup': [x,y,z] }
        """
        self.renderer.update_scene(self.d, camera=cam_name)
        
        # 1. Capture RGB
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()
        
        # 2. Capture Depth
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        
        # YOLO inference (on RGB)
        results = self.yolo(rgb, verbose=False)[0]
        annotated_frame = results.plot()
        
        results_3d = {}
        
        # Get Camera Intrinsics
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        cam_pos = self.d.cam_xpos[cam_id]
        cam_mat = self.d.cam_xmat[cam_id].reshape(3, 3)
        fovy = self.m.cam_fovy[cam_id]
        
        height, width = rgb.shape[:2]
        focal = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
        cx, cy = width / 2.0, height / 2.0

        for box in results.boxes:
            cls_name = self.yolo.names[int(box.cls[0])]
            if cls_name not in ["bottle", "cup", "vase", "glass"]:
                continue
            
            # Use center pixel of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Ensure u, v are within bounds
            u = np.clip(u, 0, width - 1)
            v = np.clip(v, 0, height - 1)
            
            # Get depth from map
            z_depth = depth[v, u]
            
            # Project to World Coords
            # ray in camera space
            rx = (u - cx) / focal
            ry = -(v - cy) / focal
            rz = -1.0
            
            ray_cam = np.array([rx, ry, rz])
            ray_cam /= np.linalg.norm(ray_cam)
            
            # ray in world space
            ray_world = cam_mat @ ray_cam
            
            world_pos = cam_pos + ray_world * z_depth
            results_3d[cls_name] = world_pos
            
            # Draw on annotated frame
            cv2.circle(annotated_frame, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"3D: {world_pos[0]:.2f},{world_pos[1]:.2f}", 
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Save debug image
        cv2.imwrite("vision_debug.jpg", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        return results_3d
