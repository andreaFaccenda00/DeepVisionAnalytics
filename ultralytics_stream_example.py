import argparse
from typing import List

import cv2
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import ClockBasedTimer

import supervision as sv

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


class CustomSink:
    def __init__(self, zone_configuration_path: str, classes: List[int]):
        self.classes = classes
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
        self.fps_monitor = sv.FPSMonitor()
        self.polygons = load_zones_config(file_path=zone_configuration_path)
        self.timers = [ClockBasedTimer() for _ in self.polygons]
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in self.polygons
        ]

    def on_prediction(self, detections: sv.Detections, frame: VideoFrame) -> None:
        self.fps_monitor.tick()
        fps = self.fps_monitor.fps

        detections = detections[find_in_list(detections.class_id, self.classes)]
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = frame.image.copy()
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=f"{fps:.1f}",
            text_anchor=sv.Point(40, 30),
            background_color=sv.Color.from_hex("#A351FB"),
            text_color=sv.Color.from_hex("#000000"),
        )

        for idx, zone in enumerate(self.zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = zone.trigger(detections=detections)
            zone.increment_occupation_count(detections_in_zone.count)
            annotated_frame = sv.draw_text(
                scene=annotated_frame,
                text=str(zone.occupation_count),
                text_anchor=sv.Point(zone.polygon[0][0], zone.polygon[0][1] - 10),
                background_color=COLORS.by_idx(idx),
                text_color=sv.Color.from_hex("#000000"),
            )

        annotated_frame = sv.draw_detections(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = LABEL_ANNOTATOR.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = COLOR_ANNOTATOR.annotate(scene=annotated_frame, detections=detections)
        
        cv2.imshow('Annotated Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

def main(rtsp_url: str, zone_configuration_path: str, weights: str, device: str, confidence: float, iou: float, classes: List[int]):
    # Load the model
    model = YOLO(weights)

    # Set up the inference pipeline
    sink = CustomSink(zone_configuration_path=zone_configuration_path, classes=classes)
    
    def inference_callback(frame: VideoFrame) -> sv.Detections:
        results = model(frame.image, device=device, conf=confidence, iou=iou)
        detections = sv.Detections.from_yolov8(results)
        return detections

    pipeline = InferencePipeline.init_with_custom_logic(
        video_reference=rtsp_url,
        on_video_frame=inference_callback,
        on_prediction=sink.on_prediction,
    )

    pipeline.start()

    try:
        pipeline.join()
    except KeyboardInterrupt:
        pipeline.terminate()


if __name__ == "__main__":
    # Directly set the parameters
    rtsp_url = 'https://www.skylinewebcams.com/it/webcam/belgique/flandres/bruges/markt.html'
    zone_configuration_path = 'data\config.json'  
    weights = 'models/best.pt'
    device = 'cuda'
    confidence_threshold = 0.3
    iou_threshold = 0.7
    classes = []

    main(
        rtsp_url=rtsp_url,
        zone_configuration_path=zone_configuration_path,
        weights=weights,
        device=device,
        confidence=confidence_threshold,
        iou=iou_threshold,
        classes=classes,
    )
