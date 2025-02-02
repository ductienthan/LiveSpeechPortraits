import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path
import argparse

class FacialLandmarkExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image):
        """Extract 3D landmarks from image using MediaPipe."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        points3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Convert to the 73-point format used in the demo
        # Note: This is a simplified mapping and might need adjustment
        indices = [
            # Face contour (17 points)
            *range(0, 17),
            # Eyebrows (10 points)
            *range(17, 27),
            # Nose (9 points)
            *range(27, 36),
            # Eyes (10 points)
            *range(36, 46),
            # Mouth (18 points)
            *range(46, 64),
            # Additional points (9 points)
            *range(64, 73)
        ]
        
        return points3d[indices]

def process_video(video_path, extractor):
    """Extract 3D landmarks from video frames."""
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(total_frames), desc=f"Processing {Path(video_path).name}"):
        ret, frame = cap.read()
        if not ret:
            break
            
        landmarks = extractor.extract_landmarks(frame)
        if landmarks is not None:
            all_landmarks.append(landmarks)
    
    cap.release()
    return np.array(all_landmarks)

def main():
    parser = argparse.ArgumentParser(description='Generate mean_pts3d.npy from videos')
    parser.add_argument('--video_dir', type=str, required=True, 
                        help='Directory containing training videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save mean_pts3d.npy')
    parser.add_argument('--video_ext', type=str, default='mp4',
                        help='Video file extension to process')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize landmark extractor
    extractor = FacialLandmarkExtractor()

    # Process all videos
    all_video_landmarks = []
    video_paths = list(Path(args.video_dir).glob(f'*.{args.video_ext}'))
    
    if not video_paths:
        print(f"No {args.video_ext} files found in {args.video_dir}")
        return

    for video_path in video_paths:
        landmarks = process_video(str(video_path), extractor)
        if len(landmarks) > 0:
            all_video_landmarks.append(landmarks)

    if not all_video_landmarks:
        print("No landmarks were successfully extracted!")
        return

    # Calculate mean 3D points
    all_landmarks = np.concatenate(all_video_landmarks, axis=0)
    mean_pts3d = np.mean(all_landmarks, axis=0)

    # Save the results
    output_path = os.path.join(args.output_dir, 'mean_pts3d.npy')
    np.save(output_path, mean_pts3d)
    print(f"Saved mean_pts3d.npy to {output_path}")
    print(f"Shape of mean_pts3d: {mean_pts3d.shape}")

    # Also save the raw points for reference
    pts3d_path = os.path.join(args.output_dir, 'pts3d.npy')
    np.save(pts3d_path, all_landmarks)
    print(f"Saved all landmarks to {pts3d_path}")
    print(f"Total frames processed: {len(all_landmarks)}")

if __name__ == "__main__":
    main() 