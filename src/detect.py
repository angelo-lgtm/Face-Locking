import cv2


CAMERA_INDEX = 0


def main():
    import argparse

    from .workflow import show

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--camera", type=int, default=CAMERA_INDEX)
    cli.add_argument("--windowed", action="store_true",
                     help="Open in a resizable window instead of fullscreen")
    args = cli.parse_args()

    detector_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(detector_path)

    if detector.empty():
        raise RuntimeError(
            f"Could not load Haar detector: {detector_path}"
        )

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Face detection running. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(70, 70),
        )

        for x, y, w, h in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            ) 

        show("Face Detection", frame, fullscreen=not args.windowed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()