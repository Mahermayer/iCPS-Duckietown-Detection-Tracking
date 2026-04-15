#!/usr/bin/env python3
"""TCP GPU server for Duckiebot remote lane perception and control."""

import math
import os
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np

import lane_pipeline as lf
from visual import viz_lane_follow

PORT = int(os.environ.get("GPU_SERVER_PORT", "5001"))
HOST = os.environ.get("GPU_SERVER_HOST", "0.0.0.0")
SHOW_GUI = os.environ.get("SHOW_GUI", "1") == "1"
SAFE_SPEED = float(os.environ.get("SAFE_SPEED", "0.1"))
SOCKET_TIMEOUT = float(os.environ.get("SERVER_SOCKET_TIMEOUT", "10.0"))

frame_qs = {}
followers = {}
frame_qs_lock = threading.Lock()


def _close_socket(conn):
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


class GpuLaneServer:
    def __init__(self):
        self.infer_lock = threading.Lock()

    def _send_reply(self, conn, vehicle, frame_id, v, omega, t_server, aoi_server=0.0):
        payload = f"{vehicle},{frame_id},{v:.3f},{omega:.3f},{t_server:.6f},{aoi_server:.6f}\n".encode("ascii")
        conn.sendall(payload)

    def _send_safe_reply(self, conn, vehicle, frame_id, last_v, last_omega):
        try:
            self._send_reply(conn, vehicle, frame_id, last_v, last_omega, time.time(), 0.0)
        except Exception as exc:
            print(f"[net] failed to send safe reply: {exc}", flush=True)

    def _get_follower(self, vehicle):
        if vehicle not in followers:
            followers[vehicle] = lf.clone_follower()
        return followers[vehicle]

    def handle_client(self, conn, addr):
        client_ip = addr[0]
        print(f"[net] connection from {addr}", flush=True)

        last_v, last_omega = 0.0, 0.0
        session_vehicle = None
        conn.settimeout(SOCKET_TIMEOUT)
        fp = conn.makefile("rb")

        try:
            while True:
                header = fp.readline()
                if not header:
                    print(f"[net] client disconnected: {addr}", flush=True)
                    break

                try:
                    header_text = header.decode("ascii").strip()
                    parts = header_text.split(",")
                    if len(parts) == 7:
                        # Older clients send a best-effort timing-stats line after
                        # each reply. This lean server does not log it.
                        continue
                    if len(parts) != 4:
                        raise ValueError(f"expected 4 fields, got {len(parts)}")
                    img_size_s, vehicle, frame_id_s, t_gen_s = parts
                    img_size = int(img_size_s)
                    frame_id = int(frame_id_s)
                    t_gen = float(t_gen_s)
                except Exception as exc:
                    print(f"[net] malformed header from {addr}: {exc}", flush=True)
                    continue

                if session_vehicle is None:
                    session_vehicle = vehicle
                    with frame_qs_lock:
                        frame_qs.setdefault(session_vehicle, deque(maxlen=2))
                    print(f"[bind] {client_ip} -> veh_name={session_vehicle}", flush=True)

                data = fp.read(img_size)
                if len(data) != img_size:
                    print(f"[net] incomplete frame {frame_id}: got {len(data)} of {img_size} bytes", flush=True)
                    self._send_safe_reply(conn, session_vehicle or vehicle, frame_id, last_v, last_omega)
                    continue

                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[net] JPEG decode failed for frame {frame_id}", flush=True)
                    self._send_safe_reply(conn, session_vehicle or vehicle, frame_id, last_v, last_omega)
                    continue

                img = cv2.resize(img, (640, 640))
                t_server = time.time()

                with self.infer_lock:
                    try:
                        err_px, lane_w_px, boxes, seg, rs_bgr, fb, mode, tracked_objs = lf.infer(img)
                        infer_error = False
                    except Exception as exc:
                        print(f"[infer] error on frame {frame_id}: {exc}", flush=True)
                        err_px, lane_w_px = math.nan, math.nan
                        boxes, seg, rs_bgr, fb, mode, tracked_objs = [], None, img, True, "error", []
                        infer_error = True

                follower = self._get_follower(session_vehicle)
                try:
                    if infer_error:
                        v, omega = SAFE_SPEED, last_omega
                    else:
                        v, omega = follower.update(
                            err_px,
                            lane_w_px,
                            boxes,
                            rs_bgr.shape[:2],
                            last_cam_bgr=rs_bgr,
                            fallback=fb,
                            tracked_objs=tracked_objs,
                        )
                except Exception as exc:
                    print(f"[ctrl] error on frame {frame_id}: {exc}", flush=True)
                    v, omega = last_v, last_omega

                last_v, last_omega = float(v), float(omega)
                aoi_server = time.time() - t_gen

                try:
                    self._send_reply(conn, session_vehicle, frame_id, last_v, last_omega, t_server, aoi_server)
                except Exception as exc:
                    print(f"[net] send error on frame {frame_id}: {exc}", flush=True)
                    break

                if SHOW_GUI:
                    try:
                        gui_frame = viz_lane_follow(
                            rs_bgr,
                            boxes,
                            seg,
                            err_px,
                            last_v,
                            last_omega,
                            mode=mode,
                            show_mode=True,
                            tracked_objs=tracked_objs,
                        )
                        if isinstance(gui_frame, np.ndarray):
                            with frame_qs_lock:
                                frame_qs.setdefault(session_vehicle, deque(maxlen=2)).append(gui_frame)
                    except Exception as exc:
                        print(f"[gui] render error: {exc}", flush=True)

        except Exception as exc:
            print(f"[net] handler exception for {addr}: {exc}", flush=True)
        finally:
            try:
                fp.close()
            except Exception:
                pass
            _close_socket(conn)
            print(f"[net] connection closed for {addr}", flush=True)

    def serve(self, host=HOST, port=PORT):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
                    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    except Exception:
                        pass
                    server_sock.bind((host, port))
                    server_sock.listen(5)
                    print(f"[server] listening on {host}:{port}", flush=True)

                    while True:
                        conn, addr = server_sock.accept()
                        threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
            except OSError as exc:
                if getattr(exc, "errno", None) == 98:
                    print(f"[server] port {port} in use; retrying in 3s", flush=True)
                else:
                    print(f"[server] socket error: {exc}; retrying in 3s", flush=True)
                time.sleep(3)
            except Exception as exc:
                print(f"[server] fatal loop error: {exc}; retrying in 3s", flush=True)
                time.sleep(3)


def run_gui_loop():
    windows = {}
    while True:
        with frame_qs_lock:
            items = [(vehicle, q[-1]) for vehicle, q in frame_qs.items() if q]

        for vehicle, frame in items:
            if vehicle not in windows:
                win_name = f"LF_{vehicle}"
                windows[vehicle] = win_name
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 960, 540)
            cv2.imshow(windows[vehicle], frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


def main():
    server = GpuLaneServer()
    threading.Thread(target=server.serve, daemon=True).start()
    print(f"[server] started; GUI={SHOW_GUI}", flush=True)

    try:
        if SHOW_GUI:
            run_gui_loop()
        else:
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("[server] interrupted", flush=True)
    finally:
        if SHOW_GUI:
            cv2.destroyAllWindows()
        print("[server] exit", flush=True)


if __name__ == "__main__":
    main()
