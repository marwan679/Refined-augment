import os
import urllib.request
import psutil
import platform
import cv2

#----------------------------------------------------------------
def check_system_specs(min_ram_gb: int = 2, min_cores: int = 2) -> bool:
    """
    Validates if the machine meets the minimum requirements to run real-time 
    AR processing smoothly.
    """
    print("--- System Compatibility Check ---")
    is_compatible = True
    
    # 1. Check RAM
    total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    ram_ok = total_ram >= min_ram_gb
    print(f"RAM: {total_ram:.2f} GB - {'[PASS]' if ram_ok else '[FAIL]'}")
    
    # 2. Check CPU Cores
    cpu_cores = psutil.cpu_count(logical=True)
    cpu_ok = cpu_cores >= min_cores
    print(f"CPU Cores: {cpu_cores} - {'[PASS]' if cpu_ok else '[FAIL]'}")
    
    # 3. Check Camera Availability
    cap = cv2.VideoCapture(0)
    camera_ok = cap.isOpened()
    if camera_ok:
        ret, _ = cap.read()
        camera_ok = ret # Confirm we can actually grab a frame
    cap.release()
    print(f"Camera Access: {'[PASS]' if camera_ok else '[FAIL]'}")
    
    # 4. OS Context (Informational)
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Final Verdict
    if not (ram_ok and cpu_ok and camera_ok):
        is_compatible = False
        print("\nWARNING: Your system might struggle to run this AR session smoothly.")
    else:
        print("\nSUCCESS: System specs meet the requirements.")
        
    return is_compatible

#----------------------------------------------------------------
def Look_for_haarcascade(filename: str = "haarcascade_frontalface_default.xml"):
    """
    Checks if the cascade file exists. If not, searches the OpenCV system 
    folder or downloads it from the official GitHub repository.
    Returns the ABSOLUTE path to the file to prevent path-resolution errors.
    """
    # 1. Check if it already exists in the current directory
    if os.path.exists(filename):
        return os.path.abspath(filename)

    # 2. Check if it exists in the OpenCV data folder (installed with pip)
    # cv2.data.haarcascades is a built-in attribute in modern OpenCV-Python
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        opencv_data_path = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(opencv_data_path):
            return os.path.abspath(opencv_data_path)

    # 3. If still not found, download it from GitHub to the current working directory
    print(f"'{filename}' not found. Downloading from official OpenCV repo...")
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return os.path.abspath(filename)
    except Exception as e:
        raise FileNotFoundError(f"Could not download or find the Haar Cascade: {e}")