from frame_extractor import extract_frames
from face_detector import detect_faces
from deepfake_model import detect_deepfake
from aggregator import aggregate_predictions

def deepfake_detection_pipeline(video_path):
    # Step 1: Extract frames from the video
    frames = extract_frames(video_path)
    
    # Step 2: Detect faces in the frames
    detected_faces = detect_faces(frames)
    
    # Step 3: Run deepfake detection on the detected faces
    predictions = []
    for faces in detected_faces:
        predictions.extend(detect_deepfake(faces))
    
    # Step 4: Aggregate predictions
    result = aggregate_predictions(predictions)
    print(result)

if __name__ == '__main__':
    video_path = 'path_to_your_video.mp4'
    deepfake_detection_pipeline(video_path)
