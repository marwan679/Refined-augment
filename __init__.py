"""
AIAugment
==========
It is a Python package that provides tools 
for enhancing and augmenting various types of data using
artificial intelligence techniques.
"""

from .Engine import apply_face_augmentation
from .Diagnostics import Look_for_haarcascade, check_system_specs

def start(image_path='AR_photo.png'):
    if check_system_specs():
        xml_path = Look_for_haarcascade()
        apply_face_augmentation(overlay_path=image_path,cascade_path=xml_path)
