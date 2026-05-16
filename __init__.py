import cv2
import numpy as np
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

class Refined_Augment:
    """
          < Refined Augment >
            ===============
    """
    
    def __init__(self):
        super().__init__()
        self.assets = []
        self.imgWarp = None
        self.maskNew = None
        self.maskInv = None
        self._last_paths = None

    def _load_overlay_images(self, overlay_paths):
        if not isinstance(overlay_paths, list):
            overlay_paths = [overlay_paths]

        self.assets = []

        for path in overlay_paths:
            asset = {
                'is_3d': False, 'img': None, 'mesh': None, 
                'vertices': None, 'triangles': None, 'normals': None
            }
            
            if isinstance(path, str):
                if path.lower().endswith(('.obj', '.stl', '.ply', '.gltf')):
                    asset['is_3d'] = True
                    
                    # 1. Robust Loading with Open3D
                    mesh = o3d.io.read_triangle_mesh(path)
                    
                    # 2. Geometry Cleanup & Normals
                    mesh.remove_duplicated_vertices()
                    mesh.remove_unreferenced_vertices()
                    mesh.compute_triangle_normals() # Crucial for 3D Shading
                    
                    # 3. Automatic Normalization (Center and scale to 0-1)
                    bbox = mesh.get_axis_aligned_bounding_box()
                    max_extent = max(bbox.get_extent())
                    if max_extent > 0:
                        mesh.scale(1.0 / max_extent, center=bbox.get_center())
                    
                    # Shift so all coordinates are strictly positive (0 to 1)
                    min_bound = mesh.get_min_bound()
                    mesh.translate(-min_bound)
                    
                    # 4. Extract to NumPy for fast OpenCV rendering
                    asset['mesh'] = mesh
                    asset['vertices'] = np.asarray(mesh.vertices)
                    asset['triangles'] = np.asarray(mesh.triangles)
                    asset['normals'] = np.asarray(mesh.triangle_normals)
                        
                elif path.startswith('http'):
                    from skimage import io
                    asset['img'] = io.imread(path)
                else:
                    asset['img'] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                asset['img'] = path

            # Standardize 2D image formats
            if not asset['is_3d'] and asset['img'] is not None:
                if asset['img'].ndim == 2:
                    asset['img'] = cv2.cvtColor(asset['img'], cv2.COLOR_GRAY2BGR)
                elif asset['img'].shape[2] == 4:
                    asset['img'] = cv2.cvtColor(asset['img'], cv2.COLOR_BGRA2BGR)
                    
            self.assets.append(asset)

    def _landmarks_to_pixels(self, landmarks, image_shape):
        height, width = image_shape[:2]
        points = []
        if hasattr(landmarks, 'landmark'):
            for lm in landmarks.landmark:
                points.append((int(lm.x * width), int(lm.y * height)))
        elif isinstance(landmarks, (list, tuple, np.ndarray)):
            for lm in landmarks:
                if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                    x, y = lm[0], lm[1]
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                        points.append((int(x * width), int(y * height)))
                    else:
                        points.append((int(x), int(y)))
        return points

    def _render_3d_object(self, imgAug, dst_points, asset):
        if not asset['is_3d'] or asset['vertices'] is None:
            return imgAug

        h_img, w_img = imgAug.shape[:2]

        x_coords, y_coords = zip(*dst_points)
        x_min, x_max = max(0, int(min(x_coords))), min(w_img, int(max(x_coords)))
        y_min, y_max = max(0, int(min(y_coords))), min(h_img, int(max(y_coords)))

        if x_max <= x_min or y_max <= y_min:
            return imgAug

        v_norm = asset['vertices']
        px = x_min + v_norm[:, 0] * (x_max - x_min)
        py = y_min + (1 - v_norm[:, 1]) * (y_max - y_min) # Flip Y
        pz = v_norm[:, 2] 
        
        px = np.clip(px, x_min, x_max)
        py = np.clip(py, y_min, y_max)
        
        projected_pts = np.column_stack((px, py)).astype(np.int32)

        # Light setup for shading (Light coming from top-front-right)
        light_dir = np.array([0.5, 0.5, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        base_color = np.array([255, 150, 0]) # BGR color (Orange/Blueish)
        ambient = 0.3

        polygons = []
        for i, face_indices in enumerate(asset['triangles']):
            if np.any(face_indices >= len(projected_pts)): continue
            
            pts = projected_pts[face_indices]
            avg_z = np.mean(pz[face_indices])
            
            # CALCULATE LIGHTING USING OPEN3D NORMALS
            normal = asset['normals'][i]
            # Dot product of face normal and light direction
            intensity = np.dot(normal, light_dir) 
            intensity = np.clip(intensity, 0, 1)
            
            # Combine ambient light with directional light
            shade_factor = ambient + (1.0 - ambient) * intensity
            color = tuple(map(int, base_color * shade_factor))

            polygons.append((avg_z, pts, color))

        # Z-SORTING (Painter's Algorithm)
        polygons.sort(key=lambda x: x[0])

        # Draw shaded faces
        for _, pts, color in polygons:
            cv2.fillPoly(imgAug, [pts], color)
            # Draw thin outline to prevent pixel gaps between polygons
            cv2.polylines(imgAug, [pts], True, color, 1)

        return imgAug

    def _overlay_on_region(self, imgAug, src_points, dst_points, asset):
        if dst_points is None:
            return imgAug
        if asset['is_3d']:
            return self._render_3d_object(imgAug, dst_points, asset)
        if src_points is None or asset['img'] is None:
            return imgAug

        h_img, w_img = imgAug.shape[:2]
        src_h, src_w = asset['img'].shape[:2]
        
        ar_object = cv2.resize(asset['img'], (src_w, src_h))
        matrix, _ = cv2.findHomography(src_points, dst_points)
        self.imgWarp = cv2.warpPerspective(ar_object, matrix, (w_img, h_img))
        
        self.maskNew = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(self.maskNew, [np.int32(dst_points)], (255, 255, 255))
        self.maskInv = cv2.bitwise_not(self.maskNew)
        
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=self.maskInv)
        imgAug = cv2.bitwise_or(self.imgWarp, imgAug)
        return imgAug

    def _compute_hand_bbox(self, points, hand_scale_factor):
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        finger_spread = 0
        if len(points) > 8:
            finger_spread = np.linalg.norm(np.array(points[4]) - np.array(points[8]))
        size = int(max(bbox_w, bbox_h, finger_spread * 1.8, 40) * hand_scale_factor)
        return int(center_x), int(center_y), size, bbox_w, bbox_h, x_min, y_min, x_max, y_max

    def _hand_overlay_points(self, x_center, y_center, size, x_min, y_min, x_max, y_max, position):
        half = size // 2
        if position == 'above':
            return np.array([[x_center - half, y_min - size], [x_center + half, y_min - size], [x_center + half, y_min], [x_center - half, y_min]], dtype=np.float32)
        elif position == 'below':
            return np.array([[x_center - half, y_max], [x_center + half, y_max], [x_center + half, y_max + size], [x_center - half, y_max + size]], dtype=np.float32)
        elif position == 'left':
            return np.array([[x_min - size, y_center - half], [x_min, y_center - half], [x_min, y_center + half], [x_min - size, y_center + half]], dtype=np.float32)
        elif position == 'right':
            return np.array([[x_max, y_center - half], [x_max + size, y_center - half], [x_max + size, y_center + half], [x_max, y_center + half]], dtype=np.float32)
        return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)

    def _apply_hand_overlays(self, imgAug, hand_landmarks_list, position, show_bounding_box, hand_scale_factor):
        if not hand_landmarks_list:
            return imgAug

        for i, hand_landmarks in enumerate(hand_landmarks_list):
            asset_idx = min(i, len(self.assets) - 1)
            asset = self.assets[asset_idx]

            points = self._landmarks_to_pixels(hand_landmarks, imgAug.shape)
            if not points: continue

            x_center, y_center, size, bbox_w, bbox_h, x_min, y_min, x_max, y_max = self._compute_hand_bbox(points, hand_scale_factor)

            if show_bounding_box:
                cv2.rectangle(imgAug, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            dst_points = self._hand_overlay_points(x_center, y_center, size, x_min, y_min, x_max, y_max, position)
            
            src_points = None
            if not asset['is_3d'] and asset['img'] is not None:
                src_h, src_w = asset['img'].shape[:2]
                src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)
                
            imgAug = self._overlay_on_region(imgAug, src_points, dst_points, asset)

        return imgAug

    def overlay(
            self, image, overlay_paths, use_haar:bool=True, manual_faces=None,
            show_bounding_box:bool=False, position:str='above', target:str='face',
            hand_landmarks=None, use_mediapipe:bool=False, hand_scale_factor:float=1.0,
        ):
            if isinstance(overlay_paths, str):
                overlay_paths = [overlay_paths]

            if self._last_paths != overlay_paths:
                self._load_overlay_images(overlay_paths)
                self._last_paths = overlay_paths 

            imgAug = image.copy()

            if target == 'hand':
                if use_mediapipe and hand_landmarks is None:
                    import mediapipe as mp
                    with mp.solutions.hands.Hands(
                        static_image_mode=False, max_num_hands=2, 
                        min_detection_confidence=0.5, min_tracking_confidence=0.5,
                    ) as hands:
                        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        hand_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []

                # ---------------------------------------------------------
                # STABILITY FIX: Sort hands from left to right on the screen
                # ---------------------------------------------------------
                if hand_landmarks and len(hand_landmarks) > 1:
                    # We sort based on the X-coordinate of the wrist (landmark 0)
                    # This ensures the left-most hand always gets asset[0]
                    # and the right-most hand always gets asset[1]
                    hand_landmarks = sorted(hand_landmarks, key=lambda hand: hand.landmark[0].x)

                return self._apply_hand_overlays(imgAug, hand_landmarks, position, show_bounding_box, hand_scale_factor)

            # ... (The rest of the face logic remains exactly the same below)
            if use_haar:
                cascade_path = Look_for_haarcascade()
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(cv2.cvtColor(imgAug, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))
            else:
                faces = manual_faces if manual_faces is not None else []

            for (x, y, w, h) in faces:
                asset = self.assets[0] 
                if show_bounding_box:
                    cv2.rectangle(imgAug, (x, y), (x+w, y+h), (255, 255, 255), 2)

                src_points = None
                if not asset['is_3d'] and asset['img'] is not None:
                    src_h, src_w = asset['img'].shape[:2]
                    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)

                if position == 'above': dst_points = np.array([[x, y - h], [x + w, y - h], [x + w, y], [x, y]], dtype=np.float32)
                elif position == 'infront': dst_points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
                
                imgAug = self._overlay_on_region(imgAug, src_points, dst_points, asset)

            return imgAug
