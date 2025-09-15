from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from models import db, Presentation, Slide, Annotation, PresentationSession, GestureLog
import cv2
import base64
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///presentation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
camera = None
detector = None
img_number = 0
annotations = [[]]
annotation_number = -1
annotation_start = False
gesture_threshold = 400  # Increased threshold for easier detection
width, height = 1280, 720
folder_path = "Presentation"
path_images = []
button_pressed = False
counter = 0
delay = 10  # Reduced delay for more responsive gestures
current_session = None
current_presentation = None
processing_thread = None
is_processing = False

def create_database():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")

def initialize_camera():
    global camera, detector, path_images
    camera = cv2.VideoCapture(0)
    camera.set(3, width)
    camera.set(4, height)
    detector = HandDetector(detectionCon=0.7, maxHands=1)  # Lower confidence for better detection
    
    # Get list of presentation images
    if os.path.exists(folder_path):
        path_images = sorted(os.listdir(folder_path), key=len)
    else:
        path_images = []
    print(f"Found {len(path_images)} images: {path_images}")

def reinitialize_camera():
    """Reinitialize camera when it fails"""
    global camera, detector
    try:
        if camera is not None:
            camera.release()
        camera = cv2.VideoCapture(0)
        camera.set(3, width)
        camera.set(4, height)
        detector = HandDetector(detectionCon=0.7, maxHands=1)  # Lower confidence for better detection
        print("Camera reinitialized successfully")
        return True
    except Exception as e:
        print(f"Failed to reinitialize camera: {e}")
        return False

def create_default_presentation():
    """Create a default presentation from images in the Presentation folder"""
    with app.app_context():
        # Check if presentations already exist
        if Presentation.query.count() > 0:
            return Presentation.query.first()
        
        # Create new presentation
        presentation = Presentation(
            title="Gesture Controlled Presentation",
            is_active=True
        )
        db.session.add(presentation)
        db.session.flush()  # Get the ID
        
        # Add slides
        for i, image_file in enumerate(path_images):
            slide = Slide(
                presentation_id=presentation.id,
                slide_number=i,
                image_path=os.path.join(folder_path, image_file)
            )
            db.session.add(slide)
        
        db.session.commit()
        print(f"Created presentation with {len(path_images)} slides")
        return presentation

def start_presentation_session(presentation_id):
    """Start a new presentation session"""
    with app.app_context():
        session = PresentationSession(
            presentation_id=presentation_id,
            session_name=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            current_slide=0,
            is_active=True
        )
        db.session.add(session)
        db.session.commit()
        return session

def log_gesture(gesture_type, gesture_data=None, session_id=None):
    """Log a gesture to the database"""
    with app.app_context():
        gesture_log = GestureLog(
            session_id=session_id,
            gesture_type=gesture_type,
            gesture_data=gesture_data
        )
        db.session.add(gesture_log)
        db.session.commit()

def save_annotation(slide_id, annotation_data, annotation_type='drawing'):
    """Save annotation to database"""
    with app.app_context():
        annotation = Annotation(
            slide_id=slide_id,
            annotation_data=annotation_data,
            annotation_type=annotation_type
        )
        db.session.add(annotation)
        db.session.commit()
        return annotation

def process_frame():
    global img_number, annotations, annotation_number, annotation_start, button_pressed, counter, current_session, current_presentation, is_processing
    
    consecutive_failures = 0
    max_failures = 10
    is_processing = True
    
    while is_processing:
        try:
            if camera is None:
                time.sleep(0.1)
                continue
                
            success, img = camera.read()
            if not success:
                consecutive_failures += 1
                print(f"Camera read failed, attempt {consecutive_failures}/{max_failures}")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive camera failures, attempting to reinitialize...")
                    reinitialize_camera()
                    consecutive_failures = 0
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            img = cv2.flip(img, 1)
            
            # Load current presentation image
            if path_images and img_number < len(path_images):
                path_full_image = os.path.join(folder_path, path_images[img_number])
                img_current = cv2.imread(path_full_image)
            else:
                img_current = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Find the hand and its landmarks
            hands, img = detector.findHands(img)
            
            # Draw Gesture Threshold line
            cv2.line(img, (0, gesture_threshold), (width, gesture_threshold), (0, 255, 0), 10)
            
            if hands and not button_pressed:
                hand = hands[0]
                cx, cy = hand["center"]
                lm_list = hand["lmList"]
                fingers = detector.fingersUp(hand)
                
                # Debug: Print finger detection
                print(f"Fingers detected: {fingers}, Hand center: ({cx}, {cy}), Threshold: {gesture_threshold}")
                
                # Constrain values for easier drawing
                x_val = int(np.interp(lm_list[8][0], [width // 2, width], [0, width]))
                y_val = int(np.interp(lm_list[8][1], [150, height-150], [0, height]))
                index_finger = (x_val, y_val)
                
                # Navigation gestures (require hand above threshold)
                if cy <= gesture_threshold:  # If hand is at the height of the face
                    if fingers == [1, 0, 0, 0, 0]:  # Left gesture
                        print("Left gesture detected - Index finger up!")
                        button_pressed = True
                        if img_number > 0:
                            img_number -= 1
                            annotations = [[]]
                            annotation_number = -1
                            annotation_start = False
                            socketio.emit('slide_changed', {'slide_number': img_number})
                            print(f"Moving to slide {img_number + 1}")
                            # Log gesture and update session
                            with app.app_context():
                                log_gesture('left', {'slide_number': img_number}, current_session.id if current_session else None)
                                if current_session:
                                    current_session.current_slide = img_number
                                    db.session.commit()
                        else:
                            print("Already at first slide")
                            
                    if fingers == [0, 0, 0, 0, 1]:  # Right gesture (pinkie finger)
                        print("Right gesture detected - Pinkie finger up!")
                        button_pressed = True
                        if img_number < len(path_images) - 1:
                            img_number += 1
                            annotations = [[]]
                            annotation_number = -1
                            annotation_start = False
                            socketio.emit('slide_changed', {'slide_number': img_number})
                            print(f"Moving to slide {img_number + 1}")
                            # Log gesture and update session
                            with app.app_context():
                                log_gesture('right', {'slide_number': img_number}, current_session.id if current_session else None)
                                if current_session:
                                    current_session.current_slide = img_number
                                    db.session.commit()
                        else:
                            print("Already at last slide")
                
                # Drawing gestures (work anywhere on screen)
                if fingers == [0, 1, 1, 0, 0]:  # Draw circle
                    cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)
                    with app.app_context():
                        log_gesture('draw_circle', {'position': index_finger}, current_session.id if current_session else None)
                
                if fingers == [0, 1, 0, 0, 0]:  # Draw line
                    if not annotation_start:
                        annotation_start = True
                        annotation_number += 1
                        annotations.append([])
                    annotations[annotation_number].append(index_finger)
                    cv2.circle(img_current, index_finger, 12, (0, 0, 255), cv2.FILLED)
                else:
                    if annotation_start and annotations and len(annotations[annotation_number]) > 1:
                        # Save annotation to database when drawing stops
                        if current_presentation and current_session:
                            with app.app_context():
                                slides = Slide.query.filter_by(presentation_id=current_presentation.id).all()
                                if img_number < len(slides):
                                    save_annotation(slides[img_number].id, annotations[annotation_number], 'drawing')
                    annotation_start = False
                
                if fingers == [0, 1, 1, 1, 0]:  # Erase last annotation
                    if annotations:
                        annotations.pop(-1)
                        annotation_number -= 1
                        button_pressed = True
                        socketio.emit('annotation_erased', {})
                        with app.app_context():
                            log_gesture('erase', {'annotation_number': annotation_number}, current_session.id if current_session else None)
            else:
                annotation_start = False
            
            if button_pressed:
                counter += 1
                if counter > delay:
                    counter = 0
                    button_pressed = False
            
            # Draw annotations
            for annotation in annotations:
                for j in range(len(annotation)):
                    if j != 0:
                        cv2.line(img_current, annotation[j - 1], annotation[j], (0, 0, 200), 12)
            
            # Resize camera feed for overlay
            img_small = cv2.resize(img, (213, 120))
            h, w, _ = img_current.shape
            img_current[0:120, w - 213: w] = img_small
            
            # Convert images to base64 for web transmission with compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Reduce quality to save bandwidth
            _, buffer_camera = cv2.imencode('.jpg', img, encode_param)
            camera_data = base64.b64encode(buffer_camera).decode('utf-8')
            
            _, buffer_slide = cv2.imencode('.jpg', img_current, encode_param)
            slide_data = base64.b64encode(buffer_slide).decode('utf-8')
            
            # Send data to frontend
            socketio.emit('frame_data', {
                'camera': camera_data,
                'slide': slide_data,
                'slide_number': img_number,
                'total_slides': len(path_images)
            })
            
            # Clear buffers to prevent memory buildup
            del buffer_camera, buffer_slide, camera_data, slide_data
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print("Too many processing errors, attempting to reinitialize camera...")
                if reinitialize_camera():
                    consecutive_failures = 0
                else:
                    print("Failed to reinitialize camera, stopping processing")
                    break
            time.sleep(0.1)
    
    print("Processing frame loop ended")
    is_processing = False

@app.route('/')
def index():
    return render_template('index.html')

# Core functionality only - no unused API routes

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    global camera, current_presentation, current_session, processing_thread, is_processing
    if camera is None:
        initialize_camera()
        # Create default presentation if none exists
        current_presentation = create_default_presentation()
        # Start a new session
        current_session = start_presentation_session(current_presentation.id)
        # Start processing thread
        is_processing = True
        processing_thread = threading.Thread(target=process_frame, daemon=True)
        processing_thread.start()
        print("Camera started and processing thread launched")
    emit('camera_started', {'status': 'success', 'presentation_id': current_presentation.id, 'session_id': current_session.id})

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera, current_session, is_processing, processing_thread
    print("Stopping camera...")
    is_processing = False
    
    if camera is not None:
        camera.release()
        camera = None
        print("Camera released")
    
    if current_session:
        with app.app_context():
            current_session.ended_at = datetime.utcnow()
            current_session.is_active = False
            db.session.commit()
        print("Session ended")
    
    # Wait for processing thread to finish
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=2.0)
        print("Processing thread stopped")
    
    emit('camera_stopped', {'status': 'success'})

if __name__ == '__main__':
    # Initialize database
    create_database()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
