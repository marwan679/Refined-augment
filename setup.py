from setuptools import setup, find_packages

setup(
    name="AIaugment",
    version="0.1.0",
    author="Marwan Gamal",
    description="A lightweight real-time AR face overlay package",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "psutil"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)