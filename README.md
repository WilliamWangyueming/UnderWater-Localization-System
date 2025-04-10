# Perception System

This module implements a real-time visual positioning system for an underwater robot and a surface vessel using AprilTag markers. It is designed to work under underwater conditions, with visual input provided via embedded video streaming hardware (e.g., USB camera connected to Raspberry Pi 5).

## Features

- 📷 **AprilTag-based relative localization** between underwater robot and surface vessel
- 🌐 **Web-based 3D visualization** of real-time positions
- 📡 **Video streaming integration** from embedded systems (e.g., MJPEG or RTSP over WiFi)
- 🎯 **Camera calibration tools** for accurate pose estimation
- 🧭 **Position filtering** and **frame smoothing** for stable and accurate tracking

## Components

| File | Description |
|------|-------------|
| `calibrate.py` | Performs intrinsic camera calibration and saves the parameters to a `.npz` file |
| `calibration_data.npz` | Resulting calibration data used for pose estimation |
| `read.py` | Reads the camera stream and runs AprilTag detection + pose estimation |
| `sample.py` | Example script demonstrating core pipeline (video + pose visualization) |
| `test_camera.py` | Simple script for verifying video input from camera |

## System Architecture

- **Underwater Robot**: Equipped with AprilTag markers
- **Surface Vessel**: Carries a downward-facing camera
- **Raspberry Pi 5**: Captures and streams video to a central processing unit
- **Central Processing Unit (Windows)**: Runs AprilTag detection, localization, and web visualization

## Visualization

A lightweight web server (based on Flask + Dash + Plotly) displays the 3D positions of each agent in real-time. AprilTag ID 0 is considered the world origin `(0,0,0)`, and all other tag poses are calculated relative to it.

## Future Work

- Multi-tag fusion for more robust localization
- Integration with underwater optical communication system
- SLAM integration for global mapping

---

**Author:** William  
**License:** MIT (or specify your license)
