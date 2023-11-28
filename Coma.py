import cv2
import os
import pafy 
import numpy as np
from pathlib import Path

from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import colors,Annotator

#Getting user input video or any
url = "https://youtu.be/EPKWu223XEg"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
best.url
videocapture = cv2.VideoCapture(best.url)   
frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

#initial value
model = YOLO('yolov8m.pt')
width = 640
height = 480
dim = (width, height)
check = 0
count = 0
color= (255,0,0)
prCount = None
track_history = defaultdict(list)

save_dir = increment_path(Path('Cam_output'))
save_dir.mkdir(parents=True)
os.chdir(save_dir)
print(os.getcwd)
video_writer = cv2.VideoWriter('VideoOut.mp4',cv2.VideoWriter_fourcc(*'MJPG'), 10, dim)

#box perimeter
counting_regions = [
    {
        'name': 'Rectangle',
        'polygon': Polygon([(-10, height), (width/2, height), (width/2, height/2), (-10, height/2)]),  # Polygon points
        'counts': 0,
        'dragging': False,
        'region_color': (0,255,200),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    } ]

#while loop for continous scanning and frame
while videocapture.isOpened():
    success, frame = videocapture.read()
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    results = model.track(frame, persist=True, show_conf=False, line_width=1)
    annotated_frame = results[0].plot(kpt_radius=2)

    # used for creating path of object
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu()
        prCount = str(len(clss))
        for box,track_ids, cls in zip(boxes,track_ids, clss):
            track = track_history[track_ids]  # Tracking Lines plot
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            annotated_frame=cv2.polylines(annotated_frame, [points], isClosed=False, color=colors(cls,True), thickness=2)

            for region in counting_regions:
                if region['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                    region['counts'] += 1

    #creating a box where it detect if only is inside it
    """
    for region in counting_regions:
        region_label = str(region['counts'])
        region_color = region['region_color']
        region_text_color = region['text_color']
        polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
        centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)

        text_size, _ = cv2.getTextSize(region_label,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.7,
                                           thickness=2)
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        annotated_frame = cv2.rectangle(annotated_frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          region_color, -1)
        annotated_frame = cv2.putText(annotated_frame, region_label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        annotated_frame = cv2.polylines(annotated_frame, [polygon_coords], isClosed=True, color=region_color, thickness=1)
   """
    for region in counting_regions:  # Reinitialize count for each region
            region['counts'] = 0
    cv2.imshow("TEST", annotated_frame)
    video_writer.write(annotated_frame)
    #quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videocapture.release()
video_writer.release()
cv2.destroyAllWindows() 