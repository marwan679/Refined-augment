# AR Face Overlay Package

This package allows you to detect faces in real-time using a webcam and overlay an image (sticker) above the detected face. It includes built-in system checks to ensure your hardware can run the processing smoothly.

Prerequisites:
The package requires Python 3.7 or higher and the following libraries:
- opencv-python
- numpy
- psutil

they will be Installed automatically using pip:
```bash
pip install opencv-python numpy psutil
```
## Usage:

You can start the application by importing the package in your main script (test.py for example).
```bash
import AI_augment as ar

# Run with your custom image
ar.start(image_path='AR_photo.png')
```
## How It Works :

- System Check: The program verifies if you have at least 2GB of RAM and 2 CPU cores to prevent lag.

- Resource Management: It automatically searches for the haarcascade_frontalface_default.xml file. If the file is not found locally or in the OpenCV system folder, it downloads it from the official repository.

- Perspective Warping: The program uses a homography matrix to scale and position the overlay image so it follows the movement of the face.

## File Descriptions :

Diagnostics.py: Contains functions to check RAM, CPU, and camera availability. It also handles the path resolution for the Haar Cascade XML file.

Engine.py: Contains the main loop that processes video frames, detects faces, and applies the image overlay logic.

init.py: Acts as the package interface, coordinating the diagnostics and the engine.

## Controls :
'q': Press the 'q' key on your keyboard to stop the video feed and close the application.

Troubleshooting
Image Load Error: Ensure the image path provided in ar.start() is correct relative to where you are running the script.

Camera Error: If the camera access fails, check if another application is using the webcam.

Persistence Error: This occurs if the XML file is corrupted or missing. The program will attempt to re-download it if you delete the existing XML file in the directory.