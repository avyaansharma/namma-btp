import time
import os
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class AccidentDetector:
    def __init__(self, iou_threshold=0.6, deceleration_threshold=12.0, speed_near_zero=1.5):
        self.iou_threshold = iou_threshold
        self.deceleration_threshold = deceleration_threshold
        self.speed_near_zero = speed_near_zero
        
        self.speed_history = {}
        self.position_history = {}
        self.flagged_accidents = set()
        
        # Keep track of active vehicle IDs currently in an accident state
        self.active_crash_vehicles = set()
        self.last_accident_time = 0
        
        # Binary Classifier integration
        self.classifier = None
        self.device = None
        self.transform = None
        
        if HAS_TORCH:
            self._load_accident_classifier()

    def _load_accident_classifier(self):
        weights_path = 'accident_classifier.pth'
        if os.path.exists(weights_path):
            try:
                print(f"[ACCIDENT DETECTOR] Loading trained classifier from {weights_path}...")
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                try:
                    self.classifier = models.resnet18(weights=None)
                except TypeError:
                    self.classifier = models.resnet18()
                    
                num_ftrs = self.classifier.fc.in_features
                self.classifier.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2)
                )
                
                state_dict = torch.load(weights_path, map_location=self.device)
                self.classifier.load_state_dict(state_dict)
                self.classifier = self.classifier.to(self.device)
                self.classifier.eval()
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                print("[ACCIDENT DETECTOR] Classifier successfully active!")
            except Exception as e:
                print(f"[ACCIDENT DETECTOR] Error loading weights: {e}")
                self.classifier = None

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        if boxAArea + boxBArea - interArea == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    def run_image_classifier(self, frame):
        if not HAS_TORCH or self.classifier is None or self.transform is None:
            return None, 0.0
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if hasattr(frame, 'shape') else frame
            pil_img = Image.fromarray(rgb_frame)
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier(tensor_img)
                probabilities = torch.softmax(outputs, dim=1)
                conf, pred_class = torch.max(probabilities, 1)
                return int(pred_class.item()), float(conf.item())
        except Exception as e:
            print(f"[ACCIDENT DETECTOR] Classifier prediction error: {e}")
            return None, 0.0

    def update_tracks(self, tracks, frame=None, current_time=None):
        if current_time is None:
            current_time = time.time()
            
        alerts = []
        active_ids = list(tracks.keys())
        
        # 1. Update speeds & centroids
        for track_id, track_data in tracks.items():
            box = track_data['box']
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            
            if track_id in self.position_history:
                prev_x, prev_y, prev_t = self.position_history[track_id]
                dt = current_time - prev_t if (current_time - prev_t) > 0 else 0.038
                
                distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                speed = distance / dt
                
                if track_id not in self.speed_history:
                    self.speed_history[track_id] = []
                self.speed_history[track_id].append(speed)
                
                if len(self.speed_history[track_id]) > 10:
                    self.speed_history[track_id].pop(0)
            
            self.position_history[track_id] = (cx, cy, current_time)

        # Cleanup lost tracks
        for track_id in list(self.position_history.keys()):
            if track_id not in active_ids:
                self.position_history.pop(track_id, None)
                self.speed_history.pop(track_id, None)
                self.active_crash_vehicles.discard(track_id)

        # Clear old active crash states after 5 seconds of no new events
        if current_time - self.last_accident_time > 5.0:
            self.active_crash_vehicles.clear()

        # 2. Heuristics overlap check (only flag if accompanied by sudden deceleration / impact)
        for i in range(len(active_ids)):
            id1 = active_ids[i]
            box1 = tracks[id1]['box']
            
            for j in range(i + 1, len(active_ids)):
                id2 = active_ids[j]
                box2 = tracks[id2]['box']
                
                iou = self._calculate_iou(box1, box2)
                if iou > self.iou_threshold:
                    # Check if there was high deceleration for either vehicle around the overlap
                    speeds1 = self.speed_history.get(id1, [])
                    speeds2 = self.speed_history.get(id2, [])
                    
                    decel1 = (np.mean(speeds1[:-1]) - speeds1[-1]) if len(speeds1) >= 3 else 0
                    decel2 = (np.mean(speeds2[:-1]) - speeds2[-1]) if len(speeds2) >= 3 else 0
                    
                    # Major crash indicator: overlap + substantial speed drop (deceleration)
                    if decel1 > self.deceleration_threshold * 10 or decel2 > self.deceleration_threshold * 10:
                        accident_pair = tuple(sorted((id1, id2)))
                        if accident_pair not in self.flagged_accidents:
                            self.flagged_accidents.add(accident_pair)
                            self.active_crash_vehicles.add(id1)
                            self.active_crash_vehicles.add(id2)
                            self.last_accident_time = current_time
                            
                            alerts.append({
                                'type': 'Accident / Crash',
                                'severity': 'Critical',
                                'vehicles': [id1, id2],
                                'description': f"Major crash detected between vehicle #{id1} and vehicle #{id2}!",
                                'timestamp': current_time
                            })
                            
        # 3. Single vehicle impact check (e.g. crashing into walls, barriers)
        for track_id in active_ids:
            speeds = self.speed_history.get(track_id, [])
            if len(speeds) >= 3:
                initial_speed = np.mean(speeds[:-1])
                final_speed = speeds[-1]
                decel = initial_speed - final_speed
                
                # High deceleration threshold to represent a real impact crash
                if decel > self.deceleration_threshold * 18 and final_speed < self.speed_near_zero * 4:
                    accident_id = (track_id,)
                    if accident_id not in self.flagged_accidents:
                        self.flagged_accidents.add(accident_id)
                        self.active_crash_vehicles.add(track_id)
                        self.last_accident_time = current_time
                        
                        alerts.append({
                            'type': 'Accident / Crash',
                            'severity': 'Critical',
                            'vehicles': [track_id],
                            'description': f"Severe single-vehicle crash detected for vehicle #{track_id}!",
                            'timestamp': current_time
                        })
                        
        # 4. Integrate Neural Classifier Prediction (Validate frame-level prediction)
        if frame is not None and self.classifier is not None:
            pred_class, conf = self.run_image_classifier(frame)
            # 0 is 'Accident' with high confidence threshold (e.g. >90%)
            if pred_class == 0 and conf > 0.90:
                accident_key = ("classifier_match", int(current_time / 5))
                if accident_key not in self.flagged_accidents:
                    self.flagged_accidents.add(accident_key)
                    self.last_accident_time = current_time
                    
                    # Associate nearby overlapping or highly decelerating vehicles with the crash state
                    for track_id in active_ids:
                        speeds = self.speed_history.get(track_id, [])
                        if len(speeds) >= 3:
                            decel = np.mean(speeds[:-1]) - speeds[-1]
                            if decel > self.deceleration_threshold * 5:
                                self.active_crash_vehicles.add(track_id)
                                
                    alerts.append({
                        'type': 'Accident / Crash',
                        'severity': 'Critical',
                        'vehicles': list(self.active_crash_vehicles),
                        'description': f"CCTV Neural network classified a major accident (confidence: {conf*100:.1f}%)",
                        'timestamp': current_time
                    })

        return alerts
