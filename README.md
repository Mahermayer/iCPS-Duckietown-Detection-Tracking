# iCPS Duckietown Detection, Tracking, and Lane Control

This repository extends the lane-segmentation training work in
[Mahermayer/Lane-Segmentation](https://github.com/Mahermayer/Lane-Segmentation)
into a live Duckiebot perception/control pipeline:

1. The vehicle ROS node captures compressed camera frames.
2. The default Duckietown launcher runs YOLO, U-Net lane segmentation, ByteTrack, and PID/FSM control on the vehicle.
3. The vehicle publishes returned `v` and `omega` commands locally.
4. The TCP GPU-server mode is still available as an alternate launcher for off-board inference.

## Main Files

- `packages/my_package/src/vehicle_client.py` - Duckiebot ROS client that streams camera frames and publishes returned commands.
- `packages/my_package/src/vehicle_local_inference.py` - Duckiebot ROS node for on-board YOLO + U-Net + ByteTrack + PID/FSM inference.
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

The default Duckietown launcher now runs all inference on the vehicle. The Docker image copies:

- `weight/`
- `lane_pipeline.py`
- `lane_constants.py`
- `segmentation/`
- `ByteTrack/`
- `packages/`

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
rosrun my_package vehicle_local_inference.py
```

Useful on-board inference environment variables:

```bash
LOCAL_FRAME_RATE=5.0
LOCAL_LOG_PERIOD=5.0
LANE_VERBOSE=0
YOLO_WEIGHTS=weight/yolo.pt
SEG_WEIGHTS=weight/segment_depthwise_se.pth
```

The local node keeps logs minimal. It reports startup once and throttles runtime status logs.

For direct on-vehicle ROS testing inside the Duckietown shell/container:

```bash
rosrun my_package vehicle_local_inference.py _frame_rate:=5.0 _log_period:=5.0
```

### Remote GPU Mode

The old TCP client is still available with the `remote_client` launcher:

```bash
dts devel run -H ruks007 -L remote_client
```

Set the GPU server address when using remote mode:

```bash
GPU_SERVER_IP=<server-ip> GPU_SERVER_PORT=5001 dts devel run -H ruks007 -L remote_client
```

For direct remote-client testing inside a Duckietown shell/container, use ROS params:

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
