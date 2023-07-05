import numpy as np
import cv2
import dlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pygame

device = torch.device('cpu')
num_classes = 4
path = 'epoch-99.pt'
classes = {'closed': 0, 'normal': 1, 'side': 2, 'yawn': 3}

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)
model.load_state_dict(torch.load(path, map_location=device))
model.eval()

detector = dlib.get_frontal_face_detector()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
])

cap = cv2.VideoCapture(0)
counter = 0
score = {label: 0 for label in classes.values()}
frames_count = {label: 0 for label in classes.values()}
required_frames = 10

pygame.mixer.init()
show_text = False

def play_alarm_sound():
    global show_text
    pygame.mixer.music.load('alarm.wav')
    pygame.mixer.music.play(-1)
    show_text = True

def stop_alarm_sound():
    pygame.mixer.music.stop()
    global show_text
    show_text = False

while True:
    ret, frame = cap.read()
    faces = detector(frame)

    try:
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            im = frame[y1:y2, x1:x2]
            pil_image = Image.fromarray(im)

            dl_frame = transform(pil_image)
            dl_frame = torch.unsqueeze(dl_frame, axis=0)
            prediction = model(dl_frame).squeeze(0).softmax(0)
            pred_label = list(classes.keys())[torch.argmax(prediction)]
            print(pred_label)
            cv2.putText(frame, 'Prediction: ' + pred_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Update scores and check for alarm trigger
            for label in classes.values():
                if label in [classes['closed'], classes['side'], classes['yawn']]:
                    if pred_label == list(classes.keys())[label]:
                        frames_count[label] += 1
                        if frames_count[label] >= required_frames:
                            score[label] += 1
                            if score[label] == 1:
                                play_alarm_sound()
                    else:
                        frames_count[label] = 0
                        score[label] = 0
                else:
                    frames_count[label] = 0
                    score[label] = 0

            if all(score[label] == 0 for label in score):
                stop_alarm_sound()

    except Exception as e:
        print(f"Error: {str(e)}")
    if show_text:
        cv2.putText(frame, "ALERT: Drowsiness Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()