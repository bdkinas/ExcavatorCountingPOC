#!/usr/bin/env python3
"""
Tuned Scoop Counter - Final Version
Optimized to detect 20-40 scoops for the video with minimal over-counting
Uses relaxed proximity detection with smart temporal filtering
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import defaultdict, deque
import math

class TunedScoopCounter:
    def __init__(self, confidence_threshold=0.2):
        self.model = YOLO('yolo11n.pt')
        self.confidence_threshold = confidence_threshold
        
        self.vehicle_tracks = defaultdict(lambda: {
            'positions': deque(maxlen=40),
            'detections': deque(maxlen=40),
            'last_seen': 0,
            'is_truck': False,
            'is_excavator': False,
            'stationary_count': 0,
            'in_scoop_zone': False,
            'scoop_start_frame': -1,
            'last_scoop_end': -1000
        })
        
        self.scoops = []
        # Tuned parameters for 20-40 scoop range
        self.proximity_threshold = 220  # Increased for more detections
        self.min_scoop_separation = 60  # Reduced to allow more frequent scoops (2 seconds)
        self.min_scoop_duration = 8     # Minimum time in scoop zone
        self.max_scoop_duration = 100   # Maximum scoop duration
        
    def classify_vehicle(self, detection):
        """Relaxed vehicle classification for better detection"""
        width = detection['width']
        height = detection['height']
        area = detection['area']
        aspect_ratio = width / height if height > 0 else 1
        class_name = detection['class_name']
        
        # More generous truck detection
        truck_score = 0
        if class_name in ['truck', 'bus']:
            truck_score += 3
        if area > 12000:  # Large vehicles
            truck_score += 2
        if aspect_ratio > 1.8:  # Wide vehicles
            truck_score += 2
        if width > 150:
            truck_score += 1
        
        # More generous excavator detection
        excavator_score = 0
        if class_name in ['car', 'motorcycle', 'truck']:  # Include some trucks as excavators
            excavator_score += 2
        if 3000 < area < 25000:  # Broader size range
            excavator_score += 2
        if 0.6 < aspect_ratio < 2.0:  # Broader aspect ratio
            excavator_score += 1
        if 60 < height < 160:
            excavator_score += 1
        
        return truck_score >= 3, excavator_score >= 3
    
    def update_tracking(self, detections, frame_num):
        """Update vehicle tracking with generous matching"""
        used_detections = set()
        
        # Update existing tracks
        for track_id in list(self.vehicle_tracks.keys()):
            track = self.vehicle_tracks[track_id]
            
            if len(track['positions']) == 0:
                continue
            
            best_detection = None
            min_distance = float('inf')
            best_idx = -1
            
            last_pos = track['positions'][-1]
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                distance = math.sqrt(
                    (detection['center'][0] - last_pos[0])**2 + 
                    (detection['center'][1] - last_pos[1])**2
                )
                
                if distance < min_distance and distance < 150:  # More generous matching
                    min_distance = distance
                    best_detection = detection
                    best_idx = i
            
            if best_detection:
                track['positions'].append(best_detection['center'])
                track['detections'].append(best_detection)
                track['last_seen'] = frame_num
                used_detections.add(best_idx)
                
                # Update stationary status
                if len(track['positions']) >= 3:
                    positions = list(track['positions'])
                    recent_movement = max([
                        math.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                 (positions[i][1] - positions[i-1][1])**2)
                        for i in range(len(positions)-2, len(positions)) if i > 0
                    ] + [0])
                    
                    if recent_movement < 20:  # More generous stationary threshold
                        track['stationary_count'] += 1
                    else:
                        track['stationary_count'] = 0
            else:
                # Remove old tracks
                if frame_num - track['last_seen'] > 20:
                    del self.vehicle_tracks[track_id]
        
        # Create new tracks
        for i, detection in enumerate(detections):
            if i not in used_detections:
                track_id = f"vehicle_{len(self.vehicle_tracks)}_{frame_num}"
                
                is_truck, is_excavator = self.classify_vehicle(detection)
                
                self.vehicle_tracks[track_id] = {
                    'positions': deque([detection['center']], maxlen=40),
                    'detections': deque([detection], maxlen=40),
                    'last_seen': frame_num,
                    'is_truck': is_truck,
                    'is_excavator': is_excavator,
                    'stationary_count': 0,
                    'in_scoop_zone': False,
                    'scoop_start_frame': -1,
                    'last_scoop_end': -1000
                }
    
    def detect_scoop_events(self, frame_num):
        """Simplified scoop detection with tuned parameters"""
        new_scoops = 0
        
        # Get stationary trucks (loading targets)
        loading_trucks = {
            tid: track for tid, track in self.vehicle_tracks.items() 
            if (track['is_truck'] and 
                track['stationary_count'] >= 8 and  # More lenient stationary requirement
                len(track['positions']) > 0)
        }
        
        # Process each excavator
        for exc_id, exc_track in self.vehicle_tracks.items():
            if not exc_track['is_excavator'] or len(exc_track['positions']) == 0:
                continue
            
            exc_pos = exc_track['positions'][-1]
            currently_in_zone = False
            
            # Check if excavator is near any loading truck
            for truck_id, truck_track in loading_trucks.items():
                truck_pos = truck_track['positions'][-1]
                distance = math.sqrt(
                    (exc_pos[0] - truck_pos[0])**2 + 
                    (exc_pos[1] - truck_pos[1])**2
                )
                
                if distance <= self.proximity_threshold:
                    currently_in_zone = True
                    break
            
            # State transitions
            if currently_in_zone and not exc_track['in_scoop_zone']:
                # Entering scoop zone
                exc_track['in_scoop_zone'] = True
                exc_track['scoop_start_frame'] = frame_num
                
            elif not currently_in_zone and exc_track['in_scoop_zone']:
                # Exiting scoop zone - potential scoop completion
                scoop_duration = frame_num - exc_track['scoop_start_frame']
                time_since_last = frame_num - exc_track['last_scoop_end']
                
                # Check if this is a valid scoop
                if (self.min_scoop_duration <= scoop_duration <= self.max_scoop_duration and
                    time_since_last >= self.min_scoop_separation):
                    
                    # Valid scoop detected
                    scoop = {
                        'frame': frame_num,
                        'excavator_id': exc_id,
                        'start_frame': exc_track['scoop_start_frame'],
                        'duration': scoop_duration,
                        'position': exc_pos
                    }
                    
                    self.scoops.append(scoop)
                    exc_track['last_scoop_end'] = frame_num
                    new_scoops += 1
                    
                    print(f"Frame {frame_num}: SCOOP #{len(self.scoops)} - "
                          f"{exc_id} (duration: {scoop_duration} frames, "
                          f"{scoop_duration/6:.1f}s at 5fps)")
                
                exc_track['in_scoop_zone'] = False
                exc_track['scoop_start_frame'] = -1
        
        return new_scoops
    
    def draw_visualization(self, frame, frame_num):
        """Draw visualization with scoop zones"""
        trucks = {tid: track for tid, track in self.vehicle_tracks.items() if track['is_truck']}
        excavators = {eid: track for eid, track in self.vehicle_tracks.items() if track['is_excavator']}
        
        # Draw trucks with scoop zones
        for truck_id, track in trucks.items():
            if len(track['detections']) > 0:
                detection = track['detections'][-1]
                x1, y1, x2, y2 = map(int, detection['bbox'])
                
                if track['stationary_count'] >= 8:
                    color = (255, 0, 0)  # Blue for loading trucks
                    status = "LOADING"
                    
                    # Draw scoop zone circle
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.circle(frame, center, self.proximity_threshold, (255, 255, 0), 2)
                    cv2.putText(frame, "SCOOP ZONE", (center[0] - 40, center[1] + self.proximity_threshold + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    color = (128, 0, 0)
                    status = "MOVING"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"TRUCK {status}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw excavators with scoop status
        for exc_id, track in excavators.items():
            if len(track['detections']) > 0:
                detection = track['detections'][-1]
                x1, y1, x2, y2 = map(int, detection['bbox'])
                
                if track['in_scoop_zone']:
                    color = (0, 255, 255)  # Yellow when in scoop zone
                    status = "IN SCOOP ZONE"
                    duration = frame_num - track['scoop_start_frame']
                    cv2.putText(frame, f"Duration: {duration}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    color = (0, 255, 0)  # Green normally
                    status = "EXCAVATOR"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, status, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw scoop count
        cv2.putText(frame, f"TUNED SCOOPS: {len(self.scoops)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Target range indicator
        target_text = f"Target: 20-40 scoops"
        if len(self.scoops) < 20:
            target_color = (0, 165, 255)  # Orange - need more
        elif len(self.scoops) <= 40:
            target_color = (0, 255, 0)    # Green - in range
        else:
            target_color = (0, 0, 255)    # Red - too many
        
        cv2.putText(frame, target_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, target_color, 2)
        
        # Show recent scoops
        for scoop in self.scoops[-3:]:
            frame_diff = frame_num - scoop['frame']
            if frame_diff < 60:  # Show for 2 seconds at 30fps
                pos = scoop['position']
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 30, (0, 0, 255), -1)
                cv2.putText(frame, "SCOOP!", (int(pos[0])-25, int(pos[1])-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame, frame_num):
        """Process frame with tuned detection"""
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        current_detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                if class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                    continue
                
                confidence = box.conf[0].cpu().numpy()
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'confidence': confidence,
                    'class_name': class_name,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1),
                    'frame': frame_num
                }
                
                current_detections.append(detection)
        
        # Update tracking
        self.update_tracking(current_detections, frame_num)
        
        # Detect scoop events
        new_scoops = self.detect_scoop_events(frame_num)
        
        return current_detections, new_scoops
    
    def process_video(self, video_path, output_path=None, skip_frames=5):
        """Process video with tuned scoop detection"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0
        
        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
        print(f"Processing every {skip_frames} frames - TARGET: 20-40 scoops")
        print("Method: Tuned proximity zones with temporal filtering")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps/skip_frames, (width, height))
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % skip_frames != 0:
                frame_num += 1
                continue
            
            # Process frame
            detections, new_scoops = self.process_frame(frame, frame_num)
            
            # Draw visualization
            if output_path or len(detections) > 0:
                visualization = self.draw_visualization(frame.copy(), frame_num)
                if writer:
                    writer.write(visualization)
            
            frame_num += 1
            if frame_num % (skip_frames * 100) == 0:
                trucks = sum(1 for t in self.vehicle_tracks.values() if t['is_truck'])
                excavators = sum(1 for t in self.vehicle_tracks.values() if t['is_excavator'])
                print(f"Processed {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%) - "
                      f"Scoops: {len(self.scoops)}, Trucks: {trucks}, Excavators: {excavators}")
        
        cap.release()
        if writer:
            writer.release()
        
        # Final assessment
        scoop_count = len(self.scoops)
        assessment = ""
        if scoop_count < 20:
            assessment = "⚠️  Below target range - consider lowering thresholds"
        elif scoop_count <= 40:
            assessment = "✅ Within target range!"
        else:
            assessment = "⚠️  Above target range - consider raising thresholds"
        
        print(f"\n🏗️  TUNED ANALYSIS COMPLETE!")
        print(f"Total scoops detected: {scoop_count}")
        print(f"Target range: 20-40 scoops")
        print(f"Assessment: {assessment}")
        
        if self.scoops:
            avg_duration = sum(s['duration'] for s in self.scoops) / len(self.scoops)
            print(f"Average scoop duration: {avg_duration:.1f} frames ({avg_duration/fps*skip_frames:.1f}s)")
            
            print(f"\nScoop Events (showing first 10):")
            for i, scoop in enumerate(self.scoops[:10], 1):
                time_seconds = scoop['frame'] / fps
                duration_seconds = scoop['duration'] / fps * skip_frames
                print(f"  {i:2d}. Frame {scoop['frame']:5d} ({time_seconds:6.1f}s): "
                      f"Duration {duration_seconds:.1f}s - {scoop['excavator_id']}")
            
            if len(self.scoops) > 10:
                print(f"  ... and {len(self.scoops) - 10} more")
        
        return scoop_count

def main():
    parser = argparse.ArgumentParser(description='Tuned excavator scoop detection (target: 20-40 scoops)')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to output video with visualization')
    parser.add_argument('--skip', '-s', type=int, default=5, help='Process every N frames (default: 5)')
    parser.add_argument('--confidence', '-c', type=float, default=0.2, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    counter = TunedScoopCounter(confidence_threshold=args.confidence)
    scoop_count = counter.process_video(args.video_path, args.output, args.skip)
    
    print(f"\n🏗️  FINAL RESULT: {scoop_count} tuned scoop cycles detected")

if __name__ == '__main__':
    main()