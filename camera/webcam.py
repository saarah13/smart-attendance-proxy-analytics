"""
camera/webcam.py
Webcam capture wrapper using OpenCV.
Supports:
  - Local USB/built-in webcam  (src = integer index, e.g. 0)
  - IP Webcam phone app        (src = base URL string,
                                e.g. "http://192.168.1.8:8080")
    Uses /shot.jpg endpoint for reliable frame-by-frame capture
    (NOT the /video MJPEG stream which is unreliable with OpenCV).

Provides a singleton VideoCapture and frame generator for MJPEG streaming.
"""
import cv2
import numpy as np
import threading
import time
import urllib.request
from typing import Union
from urllib.parse import urlparse


class WebcamStream:
    """Thread-safe webcam / IP-cam stream with singleton pattern."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, src: Union[int, str] = 0):
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance

    def __init__(self, src: Union[int, str] = 0):
        if self._initialized:
            return
        self.src = src
        self.cap = None
        self.frame = None
        self.running = False
        self._thread = None
        self._clients = 0          # reference counter for active MJPEG clients
        self._shot_url = None      # /shot.jpg URL for IP cam
        self._initialized = True

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ip_cam(self) -> bool:
        """True when the source is an IP Webcam URL."""
        return isinstance(self.src, str) and (
            self.src.startswith("http://") or self.src.startswith("https://")
        )

    def _build_shot_url(self) -> str:
        """
        Convert any IP Webcam URL variant into a /shot.jpg URL.
        e.g. http://192.168.1.8:8080          → http://192.168.1.8:8080/shot.jpg
             http://192.168.1.8:8080/video     → http://192.168.1.8:8080/shot.jpg
             https://192.168.1.8:8080/video    → http://192.168.1.8:8080/shot.jpg
        """
        url = self.src
        # Force http — IP Webcam does NOT support HTTPS
        if url.startswith("https://"):
            url = "http://" + url[len("https://"):]
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return base + "/shot.jpg"

    def start(self):
        """Start background capture thread."""
        if self.running:
            return self

        if self.is_ip_cam:
            self._shot_url = self._build_shot_url()
            print(f"[camera] IP Webcam mode — fetching frames from: {self._shot_url}")
            # Test connectivity
            try:
                resp = urllib.request.urlopen(self._shot_url, timeout=5)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                print("[camera] IP Webcam connection OK ✓")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot reach IP Webcam at: {self._shot_url}\n"
                    f"  Error: {e}\n"
                    "• Make sure your phone and PC are on the same Wi-Fi.\n"
                    "• Open the IP Webcam app, tap 'Start server'.\n"
                    "• Example: http://192.168.1.8:8080"
                )
        else:
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # only keep latest frame
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open webcam at index {self.src}")

        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        time.sleep(0.5)          # warm-up
        return self

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame = None
        self._clients = 0
        self._shot_url = None
        WebcamStream._instance = None

    def read(self):
        """Return the latest captured frame (BGR numpy array)."""
        return self.frame

    def mjpeg_generator(self, annotate_fn=None):
        """
        Yield MJPEG bytes for Flask streaming response.
        annotate_fn: optional callable(frame) -> annotated_frame
        Automatically releases the camera when the client disconnects.
        """
        with self._lock:
            self._clients += 1
        try:
            while self.running:
                frame = self.read()
                if frame is None:
                    time.sleep(0.03)
                    continue
                if annotate_fn:
                    frame = annotate_fn(frame.copy())
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                time.sleep(1 / 30)
        finally:
            # Called when the browser closes the tab / navigates away
            with self._lock:
                self._clients -= 1
                if self._clients <= 0:
                    self.stop()

    def capture_frames(self, n: int = 25):
        """
        Capture n frames for student registration.
        Returns list of BGR numpy arrays.
        """
        frames = []
        seen = 0
        while seen < n:
            frame = self.read()
            if frame is None:
                time.sleep(0.05)
                continue
            frames.append(frame.copy())
            seen += 1
            time.sleep(0.1)      # 10 FPS sample rate during registration
        return frames

    # ── Internal ──────────────────────────────────────────────────────────────

    def _grab_ip_frame(self):
        """Fetch a single JPEG frame from IP Webcam's /shot.jpg endpoint."""
        try:
            resp = urllib.request.urlopen(self._shot_url, timeout=3)
            jpg_data = resp.read()
            arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _capture_loop(self):
        if self.is_ip_cam:
            self._capture_loop_ip()
        else:
            self._capture_loop_local()

    def _capture_loop_local(self):
        """Standard OpenCV VideoCapture loop for local webcam."""
        _error_logged = False
        while self.running:
            try:
                ret, frame = self.cap.read()
                _error_logged = False
            except Exception as e:
                if not _error_logged:
                    print(f"[camera] Local cam read error: {e}")
                    _error_logged = True
                ret, frame = False, None

            if ret and frame is not None:
                self.frame = cv2.flip(frame, 1)   # mirror horizontally
                time.sleep(1 / 60)   # ~60 FPS cap — prevents buffer overrun
            else:
                time.sleep(0.03)

    def _capture_loop_ip(self):
        """HTTP snapshot loop for IP Webcam — grabs /shot.jpg as fast as possible."""
        consecutive_fails = 0
        MAX_FAILS = 30
        while self.running:
            frame = self._grab_ip_frame()
            if frame is not None:
                self.frame = frame
                consecutive_fails = 0
                # No sleep — HTTP round-trip (~30-80ms) naturally throttles to ~15-25 FPS
            else:
                consecutive_fails += 1
                if consecutive_fails == MAX_FAILS:
                    print(f"[camera] IP Webcam not responding — retrying ({self._shot_url})")
                    consecutive_fails = 0
                time.sleep(0.05)


def get_stream(src: Union[int, str] = 0) -> WebcamStream:
    """
    Return the global singleton WebcamStream (started if not already running).
    If a stream is already running with a *different* source, it is stopped first
    so a fresh instance is created with the requested source.

    Args:
        src: Camera source.
             - int  → local webcam index (0 = default built-in camera)
             - str  → IP Webcam URL (base or with path — auto-normalized)
                      e.g. "http://192.168.1.8:8080"
    """
    # If an existing singleton is running on a DIFFERENT source, stop it first
    existing = WebcamStream._instance
    if existing is not None and existing.src != src:
        try:
            existing.stop()   # clears _instance
        except Exception:
            WebcamStream._instance = None

    stream = WebcamStream(src)
    if not stream.running:
        stream.start()
    return stream
