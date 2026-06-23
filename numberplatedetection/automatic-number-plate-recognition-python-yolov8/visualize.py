import ast
import json
import cv2
import numpy as np
import pandas as pd

# Configuration parameters for stopped vehicle detection
STOP_DURATION_THRESHOLD_SEC = 10.0  # Duration (seconds) a vehicle must be stopped to be flagged
STOP_SPEED_THRESHOLD_PX_SEC = 15.0  # Max movement (pixels/second) to be considered stopped
TRAFFIC_JAM_RATIO = 0.5             # Ratio of stopped vehicles above which it's considered traffic

def analyze_stopped_vehicles(results, fps, duration_thresh, speed_thresh, jam_ratio):
    # trajectories: car_id -> {frame: (xc, yc, w)}
    trajectories = {}
    for idx, row in results.iterrows():
        car_id = int(row['car_id'])
        frame = int(row['frame_nmr'])
        bbox_str = str(row['car_bbox']).strip().replace('[', '').replace(']', '')
        parts = [float(x) for x in bbox_str.split() if x]
        if len(parts) == 4:
            x1, y1, x2, y2 = parts
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            if car_id not in trajectories:
                trajectories[car_id] = {}
            trajectories[car_id][frame] = (xc, yc, w)

    # stopped_state: car_id -> {frame: is_stopped}
    stopped_state = {}
    for car_id, traj in trajectories.items():
        stopped_state[car_id] = {}
        sorted_frames = sorted(traj.keys())
        for f in sorted_frames:
            # Find the position ~1 second (fps frames) ago, or earliest in that window
            prev_f = None
            for offset in range(int(fps), 0, -1):
                if (f - offset) in traj:
                    prev_f = f - offset
                    break
            if prev_f is None:
                prev_f = sorted_frames[0]
            
            if prev_f == f:
                stopped_state[car_id][f] = False
                continue
                
            xc_curr, yc_curr, w_curr = traj[f]
            xc_prev, yc_prev, _ = traj[prev_f]
            
            dist = ((xc_curr - xc_prev)**2 + (yc_curr - yc_prev)**2)**0.5
            dt = f - prev_f
            velocity = (dist / dt) * fps
            
            is_stopped = (velocity < speed_thresh) or (velocity < 0.05 * w_curr)
            stopped_state[car_id][f] = is_stopped

    # traffic jam detection per frame
    all_frames = sorted(results['frame_nmr'].unique())
    frame_traffic_jam = {}
    for f in all_frames:
        active_cars = [cid for cid in trajectories if f in trajectories[cid]]
        if len(active_cars) < 2:
            frame_traffic_jam[f] = False
            continue
        stopped_count = sum(1 for cid in active_cars if stopped_state[cid].get(f, False))
        ratio = stopped_count / len(active_cars)
        frame_traffic_jam[f] = (ratio > jam_ratio)

    # find cars uniquely stopped for duration_thresh consecutive seconds
    flagged_cars = set()
    required_frames = int(duration_thresh * fps)
    
    stopped_details = {}
    for car_id, traj in trajectories.items():
        sorted_frames = sorted(traj.keys())
        if len(sorted_frames) < required_frames:
            continue
            
        window_size = required_frames
        cond = []
        for f in sorted_frames:
            stopped = stopped_state[car_id].get(f, False)
            jam = frame_traffic_jam.get(f, False)
            cond.append(1 if (stopped and not jam) else 0)
            
        current_sum = sum(cond[:window_size])
        is_flagged = False
        first_flagged_frame = None
        last_flagged_frame = None
        
        if current_sum >= 0.9 * window_size:
            is_flagged = True
            first_flagged_frame = sorted_frames[0]
            last_flagged_frame = sorted_frames[window_size - 1]
            
        for idx in range(1, len(sorted_frames) - window_size + 1):
            current_sum = current_sum - cond[idx - 1] + cond[idx + window_size - 1]
            if current_sum >= 0.9 * window_size:
                if not is_flagged:
                    is_flagged = True
                    first_flagged_frame = sorted_frames[idx]
                last_flagged_frame = sorted_frames[idx + window_size - 1]
                
        if is_flagged:
            flagged_cars.add(car_id)
            stopped_details[car_id] = {
                "first_frame": first_flagged_frame,
                "last_frame": last_flagged_frame,
                "duration_frames": int(last_flagged_frame - first_flagged_frame + 1)
            }
            
    return flagged_cars, stopped_details


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


from tqdm import tqdm

results = pd.read_csv('./test_interpolated.csv')

# load video
video_path = r'C:\Users\Acer\OneDrive\Desktop\numberplatedetection\2103099-uhd_3840_2160_30fps.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(r'C:\Users\Acer\OneDrive\Desktop\numberplatedetection\out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop

# Run stopped vehicle analysis
flagged_cars, stopped_details = analyze_stopped_vehicles(
    results, 
    fps, 
    STOP_DURATION_THRESHOLD_SEC, 
    STOP_SPEED_THRESHOLD_PX_SEC, 
    TRAFFIC_JAM_RATIO
)

# Export flagged stopped vehicles data to JSON
stopped_json_data = []
for car_id in sorted(flagged_cars):
    plate = license_plate.get(car_id, {}).get('license_plate_number', '0')
    if plate == '0' or not plate:
        plate = "Unknown"
    
    details = stopped_details[car_id]
    stopped_json_data.append({
        "car_id": int(car_id),
        "license_plate": str(plate),
        "first_stopped_frame": int(details["first_frame"]),
        "last_stopped_frame": int(details["last_frame"]),
        "stopped_duration_seconds": round(details["duration_frames"] / fps, 2)
    })

json_path = r'C:\Users\Acer\OneDrive\Desktop\numberplatedetection\stopped_vehicles.json'
with open(json_path, 'w') as jf:
    json.dump({"stopped_vehicles": stopped_json_data}, jf, indent=4)
print(f"[Info] Saved {len(stopped_json_data)} stopped vehicles to {json_path}")


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="Rendering visualized video")

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        pbar.update(1)
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # draw car (Red if stopped compared to others, Green otherwise)
            car_id = int(df_.iloc[row_indx]['car_id'])
            color = (0, 0, 255) if car_id in flagged_cars else (0, 255, 0)

            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), color, 25,
                        line_length_x=200, line_length_y=200)

            # draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except:
                pass

        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

pbar.close()
out.release()
cap.release()
