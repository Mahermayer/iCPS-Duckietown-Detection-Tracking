# iCPS Duckietown Detection, Tracking, and Lane Control (GPU Server)

This project provides a **centralized perception and control system** for multiple Duckiebots using a remote GPU server.

Duckiebots send RGB camera frames to a GPU server over TCP, where the server processes them using:

- YOLO for object detection (YOLOv8n)
- Lightweight U-Net (LaneNet) for lane segmentation (https://github.com/Mahermayer/Lane-Segmentation)
- ByteTrack for vehicle tracking
- PID/FSM for control decisions

The server then sends **motion commands** (velocity `v` and angular velocity `omega`) back to each vehicle.

---

## How It Works

### The Pipeline

1. Each Duckiebot captures and compresses its camera feed
2. Frames are sent to a GPU server via TCP connection
3. The server analyzes the frames using perception models
4. The server computes and returns motion commands:
   - `v` (linear velocity)
   - `omega` (angular velocity)
5. Each Duckiebot receives and executes these commands

The system handles **multiple vehicles simultaneously** by routing frames based on unique vehicle identifiers.

---

## Project Layout

```
packages/my_package/src/
├── vehicle_client.py              # Client that runs on each Duckiebot
├── gpu_server.py                  # Server handling multiple clients  
├── lane_pipeline.py               # Perception and control logic
├── visual.py                      # Visualization tools
├── lane_constants.py              # Shared configuration
├── segmentation/                  # Lane segmentation models
├── ByteTrack/yolox/tracker/       # Multi-object tracking implementation
└── weight/                        # Pre-trained model files
```

---

## Getting Started

### Server Installation

1. Install dependencies (CUDA-enabled PyTorch recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. Run the GPU server:
   ```bash
   ./scripts/run_gpu_server.sh
   ```

3. Configure environment variables:
   ```
   GPU_SERVER_HOST=0.0.0.0
   GPU_SERVER_PORT=5001
   
   SHOW_GUI=0                              # Turn off for multiple vehicle
   LANE_VERBOSE=0
   
   SEG_WEIGHTS=weight/segment_depthwise_se.pth
   YOLO_WEIGHTS=weight/yolo.pt
   ```

### Duckiebot Setup

1. Build the client image:
   ```bash
   dts devel build -H <vehicle-name> -f
   ```
   Example: `dts devel build -H ruks007 -f`

2. Run with GPU inference:
   ```bash
   export GPU_SERVER_IP=<server_ip>
   export GPU_SERVER_PORT=5001
   export GPU_FRAME_RATE=15              # Reduce for multiple vehicles
   
   dts devel run -H <vehicle-name> -f -- \
       -e GPU_SERVER_IP \
       -e GPU_SERVER_PORT \
       -e GPU_FRAME_RATE
   ```

### Communication Protocol

---



```markdown
## System Flow with Timing

```mermaid
flowchart LR

A[Camera Capture] -->|t_gen| B[Frame + frame_id]
B --> C[JPEG Encode]
C --> D[TCP Send]

D --> E[GPU Server Receive]
E --> F[Perception Pipeline]
F --> G[YOLO + U-Net + ByteTrack]
G --> H[Control (PID/FSM)]
H -->|t_server| I[Compute v, omega]

I -->|aoi_server = t_server - t_gen| J[Send Response]
J --> K[Duckiebot Client]
K --> L[Apply Control]

### Running Multiple Vehicles

1. Start one GPU server instance
2. Launch each Duckiebot pointing to the same `GPU_SERVER_IP`
3. Set frame rate lower per vehicle: `export GPU_FRAME_RATE=5`

**Optimal Settings:**
- `SHOW_GUI=0`
- 8–10 FPS per vehicle
- Fast, stable network connection

### Network Requirements

Performance depends heavily on network quality:

**Recommended conditions:**
- Latency < 20 ms
- Jitter < 10–20 ms

---

## Key Points

- Uses **centralized GPU inference** for efficiency across multiple vehicles
- Integration of lightweight LaneNet for real-time lane segmentation
- Lightweight **TCP protocol** for low-latency control
- Scales to multiple vehicles on stable networks
- Default model weights included in the `weight/` directory
