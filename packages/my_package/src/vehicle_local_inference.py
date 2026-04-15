#!/usr/bin/env python3
"""Duckiebot on-board lane perception and control node."""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LANE_VERBOSE", "0")
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage


def _add_repo_to_path():
    repo_path = os.environ.get("DT_REPO_PATH")
    candidates = []
    if repo_path:
        candidates.append(Path(repo_path))
    candidates.append(Path(__file__).resolve().parents[2])

    for candidate in candidates:
        if (candidate / "lane_pipeline.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    raise RuntimeError("Could not find lane_pipeline.py in the Duckietown repo path")


def _param_bool(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


REPO_ROOT = _add_repo_to_path()
os.environ.setdefault("YOLO_WEIGHTS", str(REPO_ROOT / "weight" / "yolo.pt"))
os.environ.setdefault("SEG_WEIGHTS", str(REPO_ROOT / "weight" / "segment_depthwise_se.pth"))

import lane_pipeline as lf


class LocalInferenceNode(DTROS):
    def __init__(self, node_name):
        super(LocalInferenceNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        self.frame_rate = float(rospy.get_param("~frame_rate", os.environ.get("LOCAL_FRAME_RATE", "5.0")))
        self.safe_speed = float(rospy.get_param("~safe_speed", os.environ.get("SAFE_SPEED", "0.0")))
        self.log_period = float(rospy.get_param("~log_period", os.environ.get("LOCAL_LOG_PERIOD", "5.0")))
        self.publish_safe_on_error = _param_bool("~publish_safe_on_error", True)

        self._vehicle_name = os.environ.get("VEHICLE_NAME", "duckiebot")
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        self._latest_image = None
        self._frame_counter = 0
        self._last_log = 0.0
        self._last_v = 0.0
        self._last_omega = 0.0
        self._follower = lf.clone_follower()

        self.vel_pub = rospy.Publisher(
            f"/{self._vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1,
        )
        self.img_sub = rospy.Subscriber(
            self._camera_topic,
            CompressedImage,
            self._camera_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )

        rospy.on_shutdown(self._stop_bot)
        rospy.loginfo(
            "LocalInferenceNode ready vehicle=%s rate=%.1fHz weights=%s",
            self._vehicle_name,
            self.frame_rate,
            os.path.basename(os.environ["SEG_WEIGHTS"]),
        )

    def _camera_cb(self, msg):
        self._latest_image = msg

    def _decode_frame(self, msg):
        try:
            return self._bridge.compressed_imgmsg_to_cv2(msg)
        except Exception:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def publish_cmd(self, v, omega):
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = float(v)
        cmd.omega = float(omega)
        self.vel_pub.publish(cmd)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)

    def _log_status(self, frame_id, v, omega, mode, elapsed):
        now = time.time()
        if now - self._last_log < self.log_period:
            return
        self._last_log = now
        rospy.loginfo(
            "local fid=%d mode=%s v=%.3f omega=%.3f infer=%.2fs",
            frame_id,
            mode,
            v,
            omega,
            elapsed,
        )

    def run(self):
        rate = rospy.Rate(self.frame_rate)

        while not rospy.is_shutdown():
            msg = self._latest_image
            if msg is None:
                rate.sleep()
                continue

            self._frame_counter += 1
            frame_id = self._frame_counter
            start = time.time()

            try:
                frame = self._decode_frame(msg)
                if frame is None:
                    raise RuntimeError("camera decode returned empty frame")

                frame = cv2.resize(frame, (640, 640))
                err_px, lane_w_px, boxes, seg, rs_bgr, fallback, mode, tracked_objs = lf.infer(frame)
                v, omega = self._follower.update(
                    err_px,
                    lane_w_px,
                    boxes,
                    rs_bgr.shape[:2],
                    last_cam_bgr=rs_bgr,
                    fallback=fallback,
                    tracked_objs=tracked_objs,
                )
                self._last_v = float(v)
                self._last_omega = float(omega)
                self.publish_cmd(self._last_v, self._last_omega)
                self._log_status(frame_id, self._last_v, self._last_omega, mode, time.time() - start)

            except Exception as exc:
                rospy.logwarn_throttle(5.0, "local inference issue: %s", exc)
                if self.publish_safe_on_error:
                    self._last_v = self.safe_speed
                    self._last_omega = 0.0
                    self.publish_cmd(self._last_v, self._last_omega)

            rate.sleep()

        self.publish_stop()

    def _stop_bot(self):
        for _ in range(3):
            self.publish_stop()
            rospy.sleep(0.05)
        rospy.loginfo("LocalInferenceNode shutdown complete")


if __name__ == "__main__":
    node = LocalInferenceNode("local_inference_node")
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
