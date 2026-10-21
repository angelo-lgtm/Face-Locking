import time

import cv2


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():
    import argparse

    from .workflow import show

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--camera", type=int, default=CAMERA_INDEX)
    cli.add_argument("--windowed", action="store_true",
                     help="Open in a resizable window instead of fullscreen")
    args = cli.parse_args()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    previous_time = time.perf_counter()

    print("Camera running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Failed to read camera frame.")
            break

        current_time = time.perf_counter()

        elapsed = current_time - previous_time

        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        show("Camera Test", frame, fullscreen=not args.windowed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()