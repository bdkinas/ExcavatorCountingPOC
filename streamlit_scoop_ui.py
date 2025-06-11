#!/usr/bin/env python3
"""
Real-Time Scoop Counter - Streamlit Web UI
Interactive web interface for watching excavator scoop detection in real-time
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import math
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time Excavator Scoop Counter",
    page_icon="🏗️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load YOLO model (cached for performance)"""
    return YOLO('yolo11n.pt')

class StreamlitScoopCounter:
    def __init__(self):
        self.model = load_model()
        self.reset_tracking()
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.vehicle_tracks = defaultdict(lambda: {
            'positions': deque(maxlen=50),
            'detections': deque(maxlen=50),
            'last_seen': 0,
            'is_truck': False,
            'is_excavator': False,
            'stationary_count': 0
        })
        self.scoops = []
        self.recent_scoops = deque(maxlen=10)
    
    def classify_vehicle(self, detection):
        """Classify vehicle as truck or excavator"""
        width = detection['width']
        height = detection['height']
        area = detection['area']
        aspect_ratio = width / height if height > 0 else 1
        class_name = detection['class_name']
        
        truck_score = 0
        if class_name in ['truck', 'bus']:
            truck_score += 3
        if area > 12000:
            truck_score += 2
        if aspect_ratio > 1.8:
            truck_score += 2
        
        excavator_score = 0
        if class_name in ['car', 'motorcycle']:
            excavator_score += 2
        if 3000 < area < 15000:
            excavator_score += 2
        if 0.8 < aspect_ratio < 1.5:
            excavator_score += 2
        
        return truck_score >= 3, excavator_score >= 3
    
    def update_tracking(self, detections, frame_num):
        """Update vehicle tracking"""
        used_detections = set()
        
        # Match to existing tracks
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
                
                if distance < min_distance and distance < 120:
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
                        for i in range(len(positions)-3, len(positions)) if i > 0
                    ] + [0])
                    
                    if recent_movement < 15:
                        track['stationary_count'] += 1
                    else:
                        track['stationary_count'] = 0
        
        # Create new tracks
        for i, detection in enumerate(detections):
            if i not in used_detections:
                track_id = f"vehicle_{len(self.vehicle_tracks)}_{frame_num}"
                
                is_truck, is_excavator = self.classify_vehicle(detection)
                
                self.vehicle_tracks[track_id] = {
                    'positions': deque([detection['center']], maxlen=50),
                    'detections': deque([detection], maxlen=50),
                    'last_seen': frame_num,
                    'is_truck': is_truck,
                    'is_excavator': is_excavator,
                    'stationary_count': 0
                }
    
    def detect_scoops(self, frame_num, proximity_threshold):
        """Detect scoop delivery events"""
        new_scoops = 0
        
        trucks = {tid: track for tid, track in self.vehicle_tracks.items() 
                 if track['is_truck'] and len(track['positions']) > 0}
        excavators = {eid: track for eid, track in self.vehicle_tracks.items() 
                     if track['is_excavator'] and len(track['positions']) > 0}
        
        for exc_id, exc_track in excavators.items():
            exc_pos = exc_track['positions'][-1]
            
            for truck_id, truck_track in trucks.items():
                truck_pos = truck_track['positions'][-1]
                
                distance = math.sqrt(
                    (exc_pos[0] - truck_pos[0])**2 + 
                    (exc_pos[1] - truck_pos[1])**2
                )
                
                if self.is_scoop_delivery(exc_track, truck_track, distance, proximity_threshold):
                    # Check if this is a new scoop
                    is_new_scoop = True
                    for prev_scoop in self.scoops:
                        time_diff = frame_num - prev_scoop['frame']
                        if (prev_scoop['truck_id'] == truck_id and 
                            prev_scoop['excavator_id'] == exc_id and 
                            time_diff < 45):
                            is_new_scoop = False
                            break
                    
                    if is_new_scoop:
                        scoop = {
                            'frame': frame_num,
                            'truck_id': truck_id,
                            'excavator_id': exc_id,
                            'distance': distance,
                            'truck_pos': truck_pos,
                            'excavator_pos': exc_pos
                        }
                        self.scoops.append(scoop)
                        self.recent_scoops.append(scoop)
                        new_scoops += 1
        
        return new_scoops
    
    def is_scoop_delivery(self, exc_track, truck_track, distance, proximity_threshold):
        """Check if current state represents a scoop delivery"""
        if distance > proximity_threshold:
            return False
        
        if truck_track['stationary_count'] < 5:
            return False
        
        exc_pos = exc_track['positions'][-1]
        truck_detection = truck_track['detections'][-1]
        truck_bbox = truck_detection['bbox']
        
        truck_x1, truck_y1, truck_x2, truck_y2 = truck_bbox
        truck_width = truck_x2 - truck_x1
        
        bed_start = truck_x1 + truck_width * 0.4
        bed_end = truck_x2
        vertical_tolerance = 80
        
        return (bed_start <= exc_pos[0] <= bed_end + 50 and 
                truck_y1 - vertical_tolerance <= exc_pos[1] <= truck_y2 + 30)
    
    def process_frame(self, frame, frame_num, confidence_threshold, proximity_threshold):
        """Process frame for detection and tracking"""
        # Run YOLO detection
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        
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
        
        # Detect scoops
        new_scoops = self.detect_scoops(frame_num, proximity_threshold)
        
        return current_detections, new_scoops
    
    def draw_visualization(self, frame, frame_num):
        """Draw detection visualization on frame"""
        trucks = {tid: track for tid, track in self.vehicle_tracks.items() if track['is_truck']}
        excavators = {eid: track for eid, track in self.vehicle_tracks.items() if track['is_excavator']}
        
        # Draw trucks in blue
        for truck_id, track in trucks.items():
            if len(track['detections']) > 0:
                detection = track['detections'][-1]
                x1, y1, x2, y2 = map(int, detection['bbox'])
                
                color = (255, 0, 0) if track['stationary_count'] >= 5 else (128, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                status = "LOADING" if track['stationary_count'] >= 5 else "MOVING"
                cv2.putText(frame, f"TRUCK {status}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw excavators in green
        for exc_id, track in excavators.items():
            if len(track['detections']) > 0:
                detection = track['detections'][-1]
                x1, y1, x2, y2 = map(int, detection['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "EXCAVATOR", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Highlight recent scoops
        for scoop in self.recent_scoops:
            frame_diff = frame_num - scoop['frame']
            if frame_diff < 90:  # Show for 3 seconds
                truck_pos = scoop['truck_pos']
                exc_pos = scoop['excavator_pos']
                
                # Draw connection line
                cv2.line(frame, 
                        (int(exc_pos[0]), int(exc_pos[1])), 
                        (int(truck_pos[0]), int(truck_pos[1])), 
                        (0, 255, 255), 3)
                
                # Draw scoop indicator
                cv2.circle(frame, (int(truck_pos[0]), int(truck_pos[1])), 20, (0, 255, 255), -1)
                cv2.putText(frame, "SCOOP!", (int(truck_pos[0])-30, int(truck_pos[1])-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw statistics overlay
        cv2.putText(frame, f"SCOOPS: {len(self.scoops)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"TRUCKS: {len(trucks)} | EXCAVATORS: {len(excavators)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

def main():
    st.title("🏗️ Real-Time Excavator Scoop Counter")
    st.markdown("Upload a video to see real-time excavator scoop detection and counting")
    
    # Initialize session state
    if 'counter' not in st.session_state:
        st.session_state.counter = StreamlitScoopCounter()
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    # Sidebar controls
    st.sidebar.header("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    proximity_threshold = st.sidebar.slider("Proximity Threshold (px)", 50, 400, 200, 10)
    frame_skip = st.sidebar.slider("Frame Skip (for speed)", 1, 20, 5, 1)
    
    st.sidebar.header("Controls")
    if st.sidebar.button("Reset Tracking"):
        st.session_state.counter.reset_tracking()
        st.session_state.video_processed = False
        st.sidebar.success("Tracking data reset!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Upload & Processing")
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            st.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
            
            # Processing controls
            col1a, col1b = st.columns(2)
            with col1a:
                process_video = st.button("🚀 Process Video", type="primary")
            with col1b:
                show_realtime = st.checkbox("Show Real-time Processing", value=True)
            
            if process_video:
                st.session_state.counter.reset_tracking()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if show_realtime:
                    video_placeholder = st.empty()
                
                frame_num = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames for performance
                    if frame_num % frame_skip != 0:
                        frame_num += 1
                        continue
                    
                    # Process frame
                    detections, new_scoops = st.session_state.counter.process_frame(
                        frame, frame_num, confidence_threshold, proximity_threshold
                    )
                    
                    # Update progress
                    progress = frame_num / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_num}/{total_frames} - "
                                   f"Scoops: {len(st.session_state.counter.scoops)}")
                    
                    # Show real-time video (every 10th processed frame for performance)
                    if show_realtime and frame_num % (frame_skip * 10) == 0:
                        visualization = st.session_state.counter.draw_visualization(frame.copy(), frame_num)
                        visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(visualization_rgb, channels="RGB", use_column_width=True)
                    
                    frame_num += 1
                
                cap.release()
                os.unlink(video_path)  # Clean up temp file
                
                st.session_state.video_processed = True
                st.success(f"✅ Processing complete! Found {len(st.session_state.counter.scoops)} scoops")
    
    with col2:
        st.header("📊 Real-Time Statistics")
        
        if st.session_state.video_processed or len(st.session_state.counter.scoops) > 0:
            # Current statistics
            trucks = sum(1 for t in st.session_state.counter.vehicle_tracks.values() if t['is_truck'])
            excavators = sum(1 for t in st.session_state.counter.vehicle_tracks.values() if t['is_excavator'])
            
            # Metrics display
            st.metric("Total Scoops", len(st.session_state.counter.scoops))
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Trucks", trucks)
            with col2b:
                st.metric("Excavators", excavators)
            
            st.metric("Total Vehicles", len(st.session_state.counter.vehicle_tracks))
            
            # Recent scoops log
            st.subheader("📝 Recent Scoops")
            
            if st.session_state.counter.scoops:
                scoop_data = []
                for i, scoop in enumerate(st.session_state.counter.scoops[-10:], 1):  # Last 10 scoops
                    scoop_data.append({
                        "Scoop #": len(st.session_state.counter.scoops) - 10 + i,
                        "Frame": scoop['frame'],
                        "Distance": f"{scoop['distance']:.1f}px"
                    })
                
                st.dataframe(scoop_data, use_container_width=True, hide_index=True)
            else:
                st.info("No scoops detected yet")
            
            # Download results
            if st.session_state.counter.scoops:
                st.subheader("📥 Export Results")
                
                # Create CSV data
                csv_data = "Scoop_Number,Frame,Truck_ID,Excavator_ID,Distance_px\n"
                for i, scoop in enumerate(st.session_state.counter.scoops, 1):
                    csv_data += f"{i},{scoop['frame']},{scoop['truck_id']},{scoop['excavator_id']},{scoop['distance']:.1f}\n"
                
                st.download_button(
                    label="📊 Download Scoop Data (CSV)",
                    data=csv_data,
                    file_name="scoop_detection_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("Upload and process a video to see statistics")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Steps:**
        1. **Upload Video**: Select an MP4, AVI, MOV, or MKV file
        2. **Adjust Parameters**: Use the sidebar to fine-tune detection settings
        3. **Process Video**: Click "Process Video" to start detection
        4. **View Results**: Watch real-time statistics and scoop detection
        5. **Export Data**: Download CSV with all detected scoops
        
        **Detection Logic:**
        - **Trucks** (blue boxes): Larger, rectangular vehicles that become stationary for loading
        - **Excavators** (green boxes): Smaller, more square vehicles that move around
        - **Scoops** (yellow highlights): When excavator is positioned over truck bed area
        
        **Parameters:**
        - **Confidence**: Lower = more detections (may include false positives)
        - **Proximity**: Maximum distance between excavator and truck for scoop detection
        - **Frame Skip**: Higher = faster processing but may miss quick events
        """)

if __name__ == '__main__':
    main()
