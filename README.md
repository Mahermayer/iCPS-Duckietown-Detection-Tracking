# iCPS Duckietown Detection, Tracking, and Lane Control

This repository extends the lane-segmentation training work in
[Mahermayer/Lane-Segmentation](https://github.com/Mahermayer/Lane-Segmentation)
into a live Duckiebot perception/control pipeline:

1. The vehicle ROS node captures compressed camera frames.
2. The node streams JPEG frames to a GPU server over TCP.
3. The server runs YOLO, U-Net lane segmentation, and ByteTrack.
4. The server estimates lane error and computes PID/FSM velocity commands.
5. The vehicle applies returned `v` and `omega` commands.
6. The server can show a local OpenCV GUI overlay with lanes, detections, tracks, and control output.

## Main Files

- `packages/my_package/src/vehicle_client.py` - Duckiebot ROS client that streams camera frames and publishes returned commands.
- `gpu_server.py` - TCP GPU server and optional OpenCV GUI.
- `lane_pipeline.py` - YOLO + U-Net + ByteTrack + PID/FSM control logic.
- `visual.py` - GUI overlay renderer.
- `lane_constants.py` - shared class names and log columns without model imports.
- `segmentation/` - U-Net model definitions, segmentation training/evaluation scripts, and model configs.
- `ByteTrack/yolox/tracker/` - minimal vendored ByteTrack tracker code used by `lane_pipeline.py`.
- `weight/` - model weights expected by the server.

## Server Setup

Install the server dependencies in a Python environment with CUDA-capable PyTorch if available:

```bash
pip install -r requirements.txt
```

Run the GPU server:

```bash
./scripts/run_gpu_server.sh
```

Useful environment variables:

```bash
GPU_SERVER_HOST=0.0.0.0
GPU_SERVER_PORT=5001
SHOW_GUI=1
LANE_VERBOSE=0
SEG_WEIGHTS=weight/segment_depthwise_se.pth
YOLO_WEIGHTS=weight/yolo.pt
```

## Vehicle Setup

Build the Duckiebot image from the project root. Replace `ruks007` with your vehicle hostname:

```bash
dts devel build -H ruks007 -f
```

Run the image on the vehicle:

```bash
dts devel run -H ruks007
```

The default launcher runs:

```bash
rosrun my_package vehicle_client.py
```

Set the GPU server address when launching if the default is not correct:

```bash
GPU_SERVER_IP=<server-ip> GPU_SERVER_PORT=5001 dts devel run -H ruks007
```

For direct ROS testing inside a Duckietown shell/container, use ROS params:

```bash
rosrun my_package vehicle_client.py _gpu_ip:=<server-ip> _gpu_port:=5001
```

On startup, the vehicle client prints the target address before connecting:

```text
Trying to connect to GPU server at <server-ip>:5001
TCP connected to <server-ip>:5001
```

The normal build/run sequence should look like:

```bash
dts devel build -H ruks007 -f
dts devel run -H ruks007
```

The TCP protocol is line-header + JPEG payload from vehicle to server, then one command reply line from server to vehicle:

```text
client -> server: img_size,vehicle,frame_id,t_gen\n + JPEG bytes
server -> client: vehicle,frame_id,v,omega,t_server,aoi_server\n
```


## Notes

- `weight/yolo.pt` and `weight/segment_depthwise_se.pth` are the runtime weights used by default.
- Generated outputs such as `output/`, `rec/`, videos, logs, and caches are ignored by Git.
