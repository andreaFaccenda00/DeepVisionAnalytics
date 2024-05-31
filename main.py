import argparse
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

source_video_path = "data\people.mp4"
zone_configuration_path = "data\config.json"
output_video_path = "analysis\pedestrian_analytics.mp4"
weights = "models\yolov8s_pedestrian.pt"
device = "cuda"
confidence = 0.3
iou = 0.7
classes = 0

model = YOLO(weights)
tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
frames_generator = sv.get_video_frames_generator(source_video_path)

polygons = load_zones_config(file_path=zone_configuration_path)
zones = [
    sv.PolygonZone(
        polygon=polygon,
        triggering_anchors=(sv.Position.CENTER,),
    )
    for polygon in polygons
]
timers = [FPSBasedTimer(video_info.fps) for _ in zones]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, video_info.fps, (video_info.width, video_info.height))

for frame in frames_generator:
    results = model(frame, verbose=False, device=device, conf=confidence)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[find_in_list(detections.class_id, classes)]
    detections = detections.with_nms(threshold=iou)
    detections = tracker.update_with_detections(detections)

    annotated_frame = frame.copy()

    for idx, zone in enumerate(zones):
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
        )

        detections_in_zone = detections[zone.trigger(detections)]
        time_in_zone = timers[idx].tick(detections_in_zone)
        custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

        annotated_frame = COLOR_ANNOTATOR.annotate(
            scene=annotated_frame,
            detections=detections_in_zone,
            custom_color_lookup=custom_color_lookup,
        )
        labels = [
            f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
            for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
        ]
        annotated_frame = LABEL_ANNOTATOR.annotate(
            scene=annotated_frame,
            detections=detections_in_zone,
            labels=labels,
            custom_color_lookup=custom_color_lookup,
        )

    out.write(annotated_frame)

out.release()
cv2.destroyAllWindows()
