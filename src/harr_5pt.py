from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


@dataclass
class Face5pt:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray


class Haar5ptDetector:
    def __init__(
        self,
        min_size: Tuple[int, int] = (50, 50),
        scale_factor: float = 1.05,
        min_neighbors: int = 4,
        max_stale_frames: int = 5,
        debug: bool = False,
    ):
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.max_stale_frames = max_stale_frames
        self.debug = debug

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_detector.empty():
            raise RuntimeError(
                f"Could not load Haar detector: {cascade_path}"
            )

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        
        self.last_kps = None
        self.stale_frames = 0

    def _extract_keypoints(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            if self.debug:
                print("MediaPipe did not detect face landmarks")
            return None

        lm = result.multi_face_landmarks[0].landmark

        indices = [
            IDX_LEFT_EYE,
            IDX_RIGHT_EYE,
            IDX_NOSE_TIP,
            IDX_MOUTH_LEFT,
            IDX_MOUTH_RIGHT,
        ]

        points = []

        for idx in indices:
            p = lm[idx]

            points.append(
                [
                    p.x * width,
                    p.y * height,
                ]
            )

        kps = np.asarray(
            points,
            dtype=np.float32,
        )

        # Ensure consistent ordering
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]

        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(
        self,
        frame: np.ndarray,
        max_faces: int = 1,
    ) -> List[Face5pt]:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        detections = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(detections) == 0:
            return []

        # Largest face first
        detections = sorted(
            detections,
            key=lambda box: box[2] * box[3],
            reverse=True,
        )

        results = []

        for x, y, w, h in detections[:max_faces]:
            kps = self._extract_keypoints(frame)

            if kps is not None:
                self.last_kps = kps
                self.stale_frames = 0
            elif (
                self.last_kps is not None
                and self.stale_frames < self.max_stale_frames
            ):
                # Use last known keypoints only for a short grace period
                # while MediaPipe temporarily fails, so stale landmarks
                # don't linger once the face has actually moved/changed.
                self.stale_frames += 1
                kps = self.last_kps
            else:
                self.last_kps = None
                self.stale_frames = 0
                continue

            results.append(
                Face5pt(
                    x1=int(x),
                    y1=int(y),
                    x2=int(x + w),
                    y2=int(y + h),
                    kps=kps.copy(),
                )
            )

        return results

    def close(self):
        self.face_mesh.close()




        
def main():
    import argparse

    from .workflow import show

    cli = argparse.ArgumentParser(description="Haar + MediaPipe 5-point preview")
    cli.add_argument("--camera", type=int, default=0)
    cli.add_argument("--windowed", action="store_true",
                     help="Open in a resizable window instead of fullscreen")
    args = cli.parse_args()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    detector = Haar5ptDetector(debug=False)

    print("Haar + 5pt test. Press q to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        faces = detector.detect(
            frame,
            max_faces=1,
        )

        for face in faces:
            cv2.rectangle(
                frame,
                (face.x1, face.y1),
                (face.x2, face.y2),
                (0, 255, 0),
                2,
            )

            for x, y in face.kps.astype(int):
                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1,
                )

        show("Haar + 5 Point Detector", frame, fullscreen=not args.windowed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()