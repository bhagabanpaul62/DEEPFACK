from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN(keep_all=True)

def detect_faces(frames):
    detected_faces = []
    for frame in frames:
        image = Image.fromarray(frame)
        faces, _ = mtcnn(image, return_prob=True)
        if faces is not None:
            detected_faces.append(faces)
    return detected_faces
