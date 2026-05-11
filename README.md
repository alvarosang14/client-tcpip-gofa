# client-tcpip-gofa

## Overview
Main code:
- `src/main.py`: main application entry point.
- `src/camera/camera.py`: captures video, detects colored objects, computes object coordinates, and optionally responds to socket requests from the GOFA controller.
- `src/gofa_socket/gofa_socket.py`: TCP/IP client that connects to the ABB GOFA Omnicore controller.

Calibration code:
- `calibracion/muestreo_roi.py`: interactive HSV color calibrator for positive and negative ROIs.

## Quick start

1. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```

2. Run the main client:
   ```bash
   python src/main.py
   ```

3. Run with GOFA socket enabled:
   ```bash
   python src/main.py socket
   ```

4. Run the HSV calibrator:
   ```bash
   python calibracion/muestreo_roi.py
   ```

## Notes
- The camera index is configured in the code by `port_camera`. Change it if needed.
- The socket host and port are set in `src/gofa_socket/gofa_socket.py`.
- Calibration shows ROI controls and prints HSV ranges for positive and negative samples.
