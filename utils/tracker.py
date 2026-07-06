from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    score: float
    missed_frames: int = 0


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0
    return intersection / union


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed_frames: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def update(self, detections: list[dict]) -> list[Track]:
        updated_tracks: list[Track] = []
        used_detection_indexes = set()

        for track in self.tracks:
            best_index = None
            best_iou = 0.0

            for index, detection in enumerate(detections):
                if index in used_detection_indexes:
                    continue

                current_iou = _iou(track.bbox, detection["bbox"])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = index

            if best_index is not None and best_iou >= self.iou_threshold:
                detection = detections[best_index]
                track.bbox = detection["bbox"]
                track.score = detection["score"]
                track.missed_frames = 0
                updated_tracks.append(track)
                used_detection_indexes.add(best_index)
            else:
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed_frames:
                    updated_tracks.append(track)

        for index, detection in enumerate(detections):
            if index in used_detection_indexes:
                continue
            updated_tracks.append(
                Track(track_id=self.next_track_id, bbox=detection["bbox"], score=detection["score"])
            )
            self.next_track_id += 1

        self.tracks = updated_tracks
        return list(self.tracks)
