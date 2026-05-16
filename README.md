
---

# 🪄 Refined Augment

**A Next-Generation Python Engine for Real-Time Augmented Reality Overlays**

Welcome to **Refined Augment**! This powerful, lightweight Python package is designed to democratize Augmented Reality (AR) development. Whether you are building an interactive webcam filter, a virtual try-on application, or an AI-driven data augmentation pipeline, Refined Augment provides the tools to seamlessly anchor 2D graphics and 3D models to human features in real-time.

Created with a focus on performance, mathematical stability, and ease of use, this engine bypasses the need for heavy game engines (like Unity or Unreal) by running purely on Python, OpenCV, MediaPipe, and Open3D.

---

## 🌟 Core Capabilities & Features

### 1. Dual-Target Tracking System

Refined Augment natively understands human anatomy and can anchor digital assets to two primary targets:

* **Hand Tracking (Powered by MediaPipe):** Automatically detects up to two hands in the frame. It calculates a dynamic, mathematically accurate bounding box by measuring the "finger spread" (the distance between the thumb and index finger) to organically scale the augmented objects as the hand opens and closes.
* **Face Tracking (Powered by Haar Cascades):** Detects human faces using OpenCV's highly optimized Haar Cascade classifiers. It also supports manual bounding box inputs for users who prefer to use their own custom face-detection models.

### 2. Multi-Asset & Multi-Hand Support

Equip your users with different items simultaneously! By passing a list of file paths to the engine, Refined Augment will dynamically distribute them.

* **Left/Right Spatial Sorting:** AR engines often suffer from "flickering" where assets randomly swap between hands from frame to frame. We solved this by implementing a strict spatial sorting algorithm that reads the X-coordinates of the user's wrists. The left-most hand always gets the first asset, and the right-most hand always gets the second asset—guaranteeing locked-in stability.

### 3. Universal Asset Loader

The engine handles the heavy lifting of parsing and normalizing files. You can mix and match formats effortlessly:

* **3D Models:** Natively supports `.obj`, `.stl`, `.ply`, and `.gltf`.
* **2D Local Images:** Supports standard formats (`.png`, `.jpg`) and automatically handles Alpha/Transparency channels for seamless blending.
* **Web Assets:** Pass a direct HTTP URL, and the engine will fetch and decode the image on the fly using `skimage`.

### 4. Advanced 3D Rendering Engine

We built a highly optimized, custom rendering pipeline that projects 3D geometry onto a 2D OpenCV canvas without requiring complex OpenGL contexts:

* **Open3D Geometry Processing:** Every 3D model is mathematically cleaned upon loading. Duplicate vertices are removed, and the mesh is perfectly centered and scaled to fit the human bounding box, no matter the original size of the file.
* **Real-Time Shading & Lighting:** Flat polygons are a thing of the past. The engine computes mathematical triangle normals and calculates the dot-product against a virtual directional light source. Faces pointed toward the light are drawn brightly, while faces turned away are cloaked in shadow, providing true depth.
* **Z-Sorting (Painter's Algorithm):** The engine calculates the average Z-depth of every single polygon and sorts them in real-time, ensuring that the "back" of a 3D model never draws on top of the "front."
* **Vectorized Projection:** By utilizing NumPy for bulk matrix calculations, the engine processes complex 3D math simultaneously rather than point-by-point, ensuring butter-smooth FPS.

### 5. Flexible Spatial Positioning

Don't just paste an image *on* a hand—position it accurately in space. You can dictate exactly where the asset should float relative to the tracked feature:

* `'infront'`: Directly overlaid on the target.
* `'above'`: Floating over the head or fingertips.
* `'below'`, `'left'`, `'right'`: Positioned alongside the target using calculated homography matrices.

---

## 📦 Installation & Prerequisites

To run Refined Augment, you need Python 3.7+ and the following core dependencies:

```bash
pip install opencv-python numpy mediapipe open3d scikit-image

```

---

## 🚀 Quick Start Guide

Here is a complete boilerplate script to get you started with real-time, dual-hand 3D tracking using your local webcam.

```python
import cv2
from refined_augment import Refined_Augment 

# 1. Initialize the AR Engine
augmenter = Refined_Augment()

# 2. Define the assets you want to use
# The engine will assign the sword to the left hand, and the shield to the right hand.
my_assets = ['assets/sword.obj', 'assets/shield.png'] 

# 3. Start the Webcam Stream
cap = cv2.VideoCapture(0)
print("AR Engine Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Flip the frame for a natural "mirror" experience
    frame = cv2.flip(frame, 1)

    # 4. Process the frame through Refined Augment
    augmented_frame = augmenter.overlay(
        image=frame,
        overlay_paths=my_assets,      # Pass our list of mixed 2D/3D assets
        target='hand',                # Track hands
        use_mediapipe=True,           # Utilize the built-in MediaPipe tracker
        position='infront',           # Anchor objects directly on the hands
        show_bounding_box=False,      # Hide the green tracking debug squares
        hand_scale_factor=1.2         # Scale the objects up by 20%
    )

    # 5. Render the output
    cv2.imshow('Refined Augment - Live', augmented_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

---

## ⚙️ API Reference

### `Refined_Augment.overlay(...)`

The core method used to process images frame-by-frame.

**Parameters:**

* `image` *(numpy.ndarray)*: The source image/frame (BGR format).
* `overlay_paths` *(str | list)*: A single string or a list of strings pointing to local files or URLs.
* `target` *(str)*: Which feature to track. Accepts `'face'` or `'hand'`.
* `position` *(str)*: Spatial anchor. Accepts `'above'`, `'below'`, `'left'`, `'right'`, or `'infront'`.
* `use_mediapipe` *(bool)*: If True, triggers automatic hand detection.
* `hand_landmarks` *(list)*: Optional. Pass your own pre-calculated MediaPipe landmarks to save processing time if you are already running MediaPipe externally.
* `hand_scale_factor` *(float)*: Multiplier for the size of the overlaid object. Default is `1.0`.
* `use_haar` *(bool)*: If True, uses built-in Haar cascades for face detection.
* `manual_faces` *(list)*: Optional. Pass a custom list of `(x, y, w, h)` bounding boxes for face tracking.
* `show_bounding_box` *(bool)*: If True, draws debug rectangles and pivot points on the output image.

---

## 👥 Authors & Acknowledgments

**Refined Augment** was architected and developed by:

* **@Marwan Gamal** – AI/ML Engineer

Created to push the boundaries of accessible, Python-native computer vision and data augmentation.
