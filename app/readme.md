Gesture Detection AI — Assistive Interaction System for Elderly Users
This project provides an assistive interaction system designed to help elderly users and individuals with limited digital literacy interact more easily with their digital devices.
The application combines:

A physical large‑button keypad for simple, reliable input

A gesture‑recognition module powered by MediaPipe + machine learning

An accessible, minimal interface with large icons and essential functions

The goal is to provide a low‑friction, intuitive and supportive user experience, enabling seniors to communicate, access apps, navigate basic tasks, and maintain independence.

The gesture‑detection engine uses a temporal window of 3D hand landmarks, motion/posture features, and a Random Forest model trained on custom datasets. It integrates into the app to trigger high‑level actions such as opening the gallery, reading PDFs, navigating notes, controlling the calendar, or confirming UI selections.

Screenshots
Below are placeholders for the screenshots you shared; replace the path/to/... with the correct relative paths or URLs.

Keypad Interaction
(Insert your three keypad captures)

![Keypad 1](Capturas/Keypad1.PNG)
![Keypad 2](path/to/Keypad2.PNG)
![Keypad 3](path/to/Keypad3.PNG)
Gallery
![Gallery 1](path/to/GALLERY1.PNG)
![Gallery 2](path/to/GALLERY2.PNG)
![Gallery 3](path/to/GALLERY3.PNG)
Notes
![Notes 1](path/to/NOTES1.PNG)
![Notes 2](path/to/NOTES2.PNG)
PDF Viewer
![PDF 1](path/to/PDF1.PNG)
![PDF 2](path/to/PDF2.PNG)
![PDF 3](path/to/PDF3.PNG)
Calendar
![Calendar](path/to/CALENDAR.PNG)
Requirements
Windows 10/11 (tested) and a UVC‑compatible webcam

Python 3.11 (recommended)

Updated pip

Storage for datasets (gesture_data/) and models (models/)

Installation
git clone <repo-url>path/to/path/to/
cd lauzhack/app

python -m venv venv
venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
Exit environment:

deactivate
Project Structure
gesture_detection/ – core detection module

integration/ – application integration hooks

api/ – FastAPI demo server

gesture_data/ – labeled gesture samples

models/ – trained ML models

notebooks/ – dataset capture & training notebooks

tests/ – CLI demos and test utilities

Quick Start
python main_viewer.py   # Live prediction viewer
python main_app.py      # Full app with incremental dataset capture
Incremental Learning Workflow
Capture new gesture samples

Verify .pkl files in gesture_data/

Retrain:

python train_model.py
Restart app/API to load the updated model

Library Usage
from gesture_detection.detector import GestureDetector

detector = GestureDetector()
gesture, confidence, hand_detected, _ = detector.process_frame(frame)
Build Executable (optional)
python build_exe.pypath/to/