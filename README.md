# iCPS Duckietown Detection, Tracking, and Lane Control (GPU Server)

This repository implements a **real-time multi-vehicle perception and control system** for Duckiebots using a centralized GPU server.

Duckiebots stream camera frames to a GPU server over TCP. The server performs:

- YOLO (object detection)
- U-Net (lane segmentation)
- ByteTrack (tracking)
- PID/FSM (control)

and returns **control commands (`v`, `omega`)** to each vehicle.

---

## System Overview

### Multi-Vehicle GPU Inference Pipeline

1. Each Duckiebot captures compressed camera frames.
2. Frames are streamed to a GPU server via TCP.
3. The server performs perception and control.
4. The server returns:
   - `v` (linear velocity)
   - `omega` (angular velocity)
5. Each Duckiebot applies the commands locally.

The system supports **multiple vehicles concurrently**, using vehicle identifiers to separate streams.

---

## Repository Structure

packages/my_package/src/
- vehicle_client.py          # Duckiebot TCP client (streams frames)

gpu_server.py                # Multi-client GPU server  
lane_pipeline.py             # YOLO + U-Net + ByteTrack + PID/FSM control  
visual.py                    # Visualization utilities  
lane_constants.py            # Shared constants  

segmentation/                # U-Net models and training code  
ByteTrack/yolox/tracker/     # ByteTrack tracker implementation  

weight/                      # Model weights  

---

## GPU Server Setup

Install dependencies (CUDA-enabled PyTorch recommended):

```bash
pip install -r requirements.txt


Run the GPU server:

./scripts/run_gpu_server.sh


Environment Variables
GPU_SERVER_HOST=0.0.0.0
GPU_SERVER_PORT=5001

SHOW_GUI=0          # Disable for multi-vehicle runs
LANE_VERBOSE=0

SEG_WEIGHTS=weight/segment_depthwise_se.pth
YOLO_WEIGHTS=weight/yolo.pt


Duckiebot Setup
Build
dts devel build -H <vehicle-name> -f

Example:

dts devel build -H ruks007 -f
Run (GPU Inference Mode)

Set GPU server parameters:

export GPU_SERVER_IP=<server_ip>
export GPU_SERVER_PORT=5001
export GPU_FRAME_RATE=15   # Reduce for multiple vehicles (8–10 recommended)

Run the client:

dts devel run -H <vehicle-name> -f -- \
    -e GPU_SERVER_IP \
    -e GPU_SERVER_PORT \
    -e GPU_FRAME_RATE
Communication Protocol

Client → Server:
img_size,vehicle,frame_id,t_gen\n + JPEG bytes

Server → Client:
vehicle,frame_id,v,omega,t_server,aoi_server\n

Images are transmitted as JPEG payloads
Server returns only control commands (v, omega)
vehicle field enables multi-vehicle routing
Multi-Vehicle Usage

To run multiple Duckiebots:

Start a single GPU server
Launch each vehicle with the same GPU_SERVER_IP
Reduce per-vehicle frame rate:
export GPU_FRAME_RATE=8
Recommended Settings
SHOW_GUI=0
FPS: 8–10 per vehicle
Strong and stable Wi-Fi connection
Performance Notes
System performance is primarily network-bound
High latency or jitter leads to:
delayed control
connection resets
Recommended network conditions:
latency < 20 ms
jitter < 10–20 ms
packet loss ≈ 0%
Notes
Default weights:
weight/yolo.pt
weight/segment_depthwise_se.pth
Generated outputs (output/, logs, caches) are ignored via .gitignore
Summary
Centralized GPU inference for multiple Duckiebots
Lightweight TCP protocol for real-time control
Scalable to multiple vehicles under stable network conditions
