"""Shared command-line arguments and webcam lifetime management."""
import argparse
from contextlib import contextmanager
import cv2
from .config import DATABASE, DETECTOR, EMBEDDER
from .embed import ArcFace
from .landmarks import FaceDetector


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--camera", type=int, default=0)
    result.add_argument("--detector", default=str(DETECTOR))
    result.add_argument("--model", default=str(EMBEDDER))
    result.add_argument("--db", default=str(DATABASE))
    result.add_argument("--preprocessing", choices=("raw", "normalized"), default="raw")
    result.add_argument("--windowed", action="store_true",
                        help="Open the preview in a resizable window instead of fullscreen")
    return result


def models(args):
    return FaceDetector(args.detector), ArcFace(args.model, args.preprocessing)
    

@contextmanager
def camera(index):
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}; close other camera apps or try --camera 1")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Camera stopped returning frames")
    # Mirror the frame horizontally for natural selfie-style viewing
    return cv2.flip(frame, 1)


def exiting(window, key):
    return key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


_WINDOWS = {}


def _screen_size():
    """Primary display size in pixels, or ``None`` when it cannot be read."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        return (int(width), int(height)) if width and height else None
    except Exception:
        return None


def show(window, frame, fullscreen=True):
    """Display ``frame`` in ``window``, filling the screen unless disabled.

    The window is created once as a normal (resizable) window, switched to
    fullscreen and then fitted to the display with the aspect ratio preserved,
    so wide views are letterboxed instead of stretched. Repeat calls only
    push pixels; geometry is recomputed if the frame size changes.
    """
    shape = tuple(frame.shape[:2])
    state = _WINDOWS.get(window)
    if state != (shape, fullscreen):
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            screen = _screen_size()
            if screen:
                scale = min(screen[0] / shape[1], screen[1] / shape[0])
                cv2.resizeWindow(window, max(1, int(shape[1] * scale)),
                                 max(1, int(shape[0] * scale)))
        else:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, 0)
        _WINDOWS[window] = (shape, fullscreen)
    cv2.imshow(window, frame)


def inset(frame, image, label=""):
    """Blit ``image`` into the bottom-right corner of ``frame``.

    Replaces the second preview window that would otherwise be hidden behind a
    fullscreen window.
    """
    height, width = image.shape[:2]
    target = max(96, min(frame.shape[1] // 4, frame.shape[0] // 3))
    factor = target / max(width, height)
    scaled_w, scaled_h = max(1, int(round(width * factor))), max(1, int(round(height * factor)))
    preview = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    x, y = frame.shape[1] - scaled_w - 8, frame.shape[0] - scaled_h - 8
    frame[y:y + scaled_h, x:x + scaled_w] = preview
    cv2.rectangle(frame, (x - 2, y - 2), (x + scaled_w + 1, y + scaled_h + 1),
                  (255, 255, 255), 2)
    if label:
        cv2.putText(frame, label, (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run(main):
    opencv_errors = (cv2.error,) if hasattr(cv2, 'error') else ()
    try:
        if not all(hasattr(cv2, name) for name in ('error', 'FaceDetectorYN', 'VideoCapture')):
            raise RuntimeError(
                'OpenCV installation is incomplete. Run: python -m pip install '
                '--force-reinstall --no-deps opencv-contrib-python==4.11.0.86')
        main()
    except (ValueError, RuntimeError, OSError) + opencv_errors as error:
        raise SystemExit(f"FaceX: {error}") from error
    except KeyboardInterrupt:
        print("\nStopped.")
