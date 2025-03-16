# app.py
import os
import cv2
import numpy as np
import face_recognition
import threading
import time
import pickle
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FACES_FOLDER'] = 'static/faces'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACES_FOLDER'], exist_ok=True)

# Global variables
known_face_encodings = []
known_face_names = []
camera = None
camera_thread = None
processing_frame = False
current_frame = None
model_trained = False
model_path = 'face_model.pkl'
camera_running = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    global known_face_encodings, known_face_names, model_trained
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
            model_trained = True if known_face_names else False
        return True
    return False

def save_model():
    global known_face_encodings, known_face_names
    with open(model_path, 'wb') as f:
        pickle.dump({
            'encodings': known_face_encodings,
            'names': known_face_names
        }, f)

def train_from_directory():
    global known_face_encodings, known_face_names, model_trained
    known_face_encodings = []
    known_face_names = []
    
    for person_name in os.listdir(app.config['FACES_FOLDER']):
        person_dir = os.path.join(app.config['FACES_FOLDER'], person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_file in os.listdir(person_dir):
            if not allowed_file(img_file):
                continue
                
            img_path = os.path.join(person_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) > 0:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
    
    model_trained = True if known_face_names else False
    save_model()
    return len(known_face_names)

def process_frame(frame):
    global known_face_encodings, known_face_names
    
    # Resize frame to speed up processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all faces in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    confidence_scores = []
    
    for face_encoding in face_encodings:
        # Compare face with known faces
        matches = []
        confidence = 0
        name = "Unknown"
        
        if known_face_encodings:
            # Calculate face distance (lower means more similar)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                # If confidence above 0.6, consider it a match
                if confidence > 0.6:
                    name = known_face_names[best_match_index]
        
        face_names.append(name)
        confidence_scores.append(confidence)
    
    # Display results on the frame
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidence_scores):
        # Scale back up face locations since the frame was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Choose color based on recognition (green for known, red for unknown)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with the name and confidence below the face
        conf_text = f"{confidence:.2f}" if confidence > 0 else "N/A"
        label_text = f"{name} ({conf_text})"
        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label_text, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def camera_loop():
    global camera, current_frame, processing_frame, camera_running
    
    while camera_running:
        if camera is None:
            time.sleep(0.1)
            continue
            
        ret, frame = camera.read()
        if not ret:
            break
            
        if not processing_frame:
            processing_frame = True
            # Process the frame with facial recognition
            processed_frame = process_frame(frame.copy())
            current_frame = processed_frame
            processing_frame = False
    
    if camera is not None:
        camera.release()

def gen_frames():
    global current_frame, camera_running
    
    while camera_running:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, generate a placeholder
            blank_image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_image, "Camera starting...", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_image)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05)  # Limit frame rate to avoid overwhelming the browser

@app.route('/')
def index():
    return render_template('index.html', model_trained=model_trained)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, camera_thread, camera_running
    
    camera_id = request.form.get('camera_id', 0)
    
    try:
        camera_id = int(camera_id)
    except ValueError:
        camera_id = 0
    
    if camera is not None:
        camera.release()
    
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        return jsonify({'success': False, 'message': f"Failed to open camera {camera_id}"})
    
    # Start camera thread if not already running
    if camera_thread is None or not camera_thread.is_alive():
        camera_running = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.daemon = True
        camera_thread.start()
    
    return jsonify({'success': True})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, camera_running
    
    camera_running = False
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({'success': True})

@app.route('/list_cameras', methods=['GET'])
def list_cameras():
    # Test the first 5 camera indices (could be extended)
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return jsonify({'cameras': available_cameras})

@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'face_image' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
        
    file = request.files['face_image']
    person_name = request.form.get('person_name', '').strip()
    
    if file.filename == '' or not person_name:
        return jsonify({'success': False, 'message': 'No file selected or no name provided'})
        
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'File type not allowed'})
    
    # Create directory for person if it doesn't exist
    person_dir = os.path.join(app.config['FACES_FOLDER'], person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(person_dir, filename)
    file.save(file_path)
    
    # Verify the image contains a face
    try:
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            os.remove(file_path)
            return jsonify({'success': False, 'message': 'No face detected in the image'})
            
    except Exception as e:
        os.remove(file_path)
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'})
    
    return jsonify({'success': True})

@app.route('/train_model', methods=['POST'])
def train_model():
    global model_trained
    
    try:
        face_count = train_from_directory()
        if face_count > 0:
            return jsonify({'success': True, 'message': f'Model trained with {face_count} face(s)'})
        else:
            return jsonify({'success': False, 'message': 'No faces found for training'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error training model: {str(e)}'})

@app.route('/get_people', methods=['GET'])
def get_people():
    people = []
    
    if os.path.exists(app.config['FACES_FOLDER']):
        for person_name in os.listdir(app.config['FACES_FOLDER']):
            person_dir = os.path.join(app.config['FACES_FOLDER'], person_name)
            if os.path.isdir(person_dir):
                # Count images for this person
                image_count = len([f for f in os.listdir(person_dir) if allowed_file(f)])
                people.append({
                    'name': person_name,
                    'image_count': image_count
                })
    
    return jsonify({'people': people})

@app.route('/delete_person', methods=['POST'])
def delete_person():
    person_name = request.form.get('person_name', '').strip()
    
    if not person_name:
        return jsonify({'success': False, 'message': 'No name provided'})
    
    person_dir = os.path.join(app.config['FACES_FOLDER'], person_name)
    
    if not os.path.isdir(person_dir):
        return jsonify({'success': False, 'message': 'Person not found'})
    
    try:
        # Delete all files in the directory
        for file in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, file))
        
        # Remove the directory
        os.rmdir(person_dir)
        
        # Retrain the model
        train_from_directory()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting person: {str(e)}'})

# Load the model when the app starts
load_model()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
