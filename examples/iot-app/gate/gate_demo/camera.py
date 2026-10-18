"""Continuous V4L2 capture owned by the example, with a shared live video stream."""

import threading
import time
from pathlib import Path

from dtr import RuntimeFault

from .core import Frame, camera_covered, camera_frame


class VideoCamera:
    def __init__(self, *, device=None, stream_url=None, width=1280, height=720, fps=15, rotate=0):
        self.device, self.stream_url = device, stream_url
        self.width, self.height, self.fps = width, height, fps
        self.rotate = rotate
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = None
        self.latest = None
        self.frames = 0
        self.started = None
        self.error = None
        self.covered = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._capture, name="brio-video", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=6)

    def _capture(self):
        capture = None
        self.started = time.monotonic()
        try:
            if not self.stream_url:
                import cv2

                if self.device is None:
                    devices = sorted(Path("/dev/v4l/by-id").glob("*video-index0"))
                    devices.sort(key=lambda item: "brio" not in item.name.lower())
                    if not devices:
                        raise RuntimeFault("camera_unavailable", "No USB camera found; connect the brio")
                    self.device = str(devices[0])
                capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                capture.set(cv2.CAP_PROP_FPS, self.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    raise RuntimeFault("camera_unavailable", "The brio is unavailable or another process is using it")
            while not self.stop_event.is_set():
                began = time.monotonic()
                if self.stream_url:
                    frame = camera_frame(self.stream_url)
                else:
                    ok, pixels = capture.read()
                    if not ok:
                        raise RuntimeFault("camera_disconnected", "The brio stopped returning video frames")
                    captured_at = time.monotonic()
                    if self.rotate:
                        pixels = cv2.rotate(pixels, {90: cv2.ROTATE_90_CLOCKWISE,
                                                    180: cv2.ROTATE_180,
                                                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotate])
                    ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if not ok:
                        raise RuntimeFault("camera_encoding", "Could not encode the captured frame")
                    frame = Frame(jpeg.tobytes(), "camera", "image/jpeg", captured_at)
                covered = camera_covered(frame) if self.frames % 15 == 0 else self.covered
                with self.condition:
                    self.latest, self.covered = frame, covered
                    self.frames += 1
                    self.condition.notify_all()
                self.stop_event.wait(max(0, 1 / self.fps - (time.monotonic() - began)))
        except Exception as exc:
            with self.condition:
                self.error = {"code": getattr(exc, "code", "camera_unavailable"), "message": str(exc)}
                self.condition.notify_all()
        finally:
            if capture is not None:
                capture.release()

    def status(self):
        with self.condition:
            elapsed = time.monotonic() - self.started if self.started else 0
            age = time.monotonic() - self.latest.captured_at if self.latest else None
            return {"device": self.device, "source": "mjpeg" if self.stream_url else "v4l2",
                    "rotation": self.rotate if not self.stream_url else 0,
                    "frames": self.frames, "fps": round(self.frames / elapsed, 2) if elapsed else 0,
                    "frame_age_ms": round(age * 1000, 1) if age is not None else None,
                    "covered": self.covered, "error": self.error,
                    "running": bool(self.thread and self.thread.is_alive() and not self.error)}

    def frame(self):
        with self.condition:
            if self.error:
                raise RuntimeFault(self.error["code"], self.error["message"])
            if self.latest is None or time.monotonic() - self.latest.captured_at > 2:
                raise RuntimeFault("camera_unavailable", "No fresh video frame is available")
            return self.latest

    def stream(self):
        sequence = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda previous=sequence: self.frames != previous or self.error
                                        or self.stop_event.is_set(), timeout=2)
                if self.error or self.stop_event.is_set():
                    return
                if self.latest is None or self.frames == sequence:
                    continue
                sequence, frame = self.frames, self.latest
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(frame.data)).encode() + b"\r\nX-Frame-Number: "
                   + str(sequence).encode() + b"\r\n\r\n" + frame.data + b"\r\n")
