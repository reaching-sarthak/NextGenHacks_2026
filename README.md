# Municipal Road Inspection AI

> An on-premise computer vision system for automatically identifying road damage from government vehicle imagery.

## 🚧 Overview

Municipalities rely heavily on manual inspections to identify potholes, cracks, and other road damage. This process is time-consuming, expensive, and difficult to scale when inspection vehicles are already collecting large amounts of road imagery.

**Municipal Road Inspection AI** is a prototype designed to automate this process.

Government inspection vehicles can capture road footage, which can then be converted into individual road images and passed through a compact convolutional neural network (CNN). The model classifies each image into one of four categories:

- **Normal** — no recognised road damage
- **Crack** — one or more recognised cracks
- **Pothole** — one or more potholes
- **Both** — cracks and potholes present together

Detected damage can then be flagged for municipal staff, allowing human inspectors to focus their attention on areas that require intervention.

---

## 🔐 Why Local AI?

A major design principle of this project is **data sovereignty**.

Government road imagery can contain sensitive information about public infrastructure, locations, vehicles, and surrounding environments. Instead of sending this data to external AI providers, this system is designed around a **small, custom-trained model that can be deployed on municipal infrastructure**.

The goal is:

```text
Government Vehicle
       ↓
Government Infrastructure
       ↓
Road Inspection Model
       ↓
Government Database
       ↓
Municipal Staff

No external AI API is required for inference.

This allows municipalities to maintain control over their inspection data while still benefiting from automated computer vision.

🧠 Machine Learning Model

The prototype uses a custom convolutional neural network trained from scratch rather than relying on YOLO, a large pretrained vision model, or an external computer vision API.

The architecture is inspired by early CNN architectures such as AlexNet while incorporating more modern components.

Model pipeline
Input Image
     ↓
Preprocessing
     ↓
Max Pooling
     ↓
Convolutional Layers
     ↓
SiLU Activation
     ↓
Residual Block
     ↓
Global Average Pooling
     ↓
Fully Connected Layer
     ↓
Dropout
     ↓
4-Class Output

The model produces probabilities for the four classification categories.

Important: Model confidence is not currently interpreted as damage severity. A probability such as 90% means the model is highly confident in its classification, not that the road damage is 90% severe.

📊 Dataset

The current prototype uses the Indian subset of the Road Damage Dataset (RDD2022).

The dataset provides Pascal VOC XML annotations containing road-damage labels and bounding boxes.

The relevant labels are mapped into the project's four classification categories.

Label mapping
RDD2022 Label	Project Classification
D00	Crack
D01	Crack
D10	Crack
D11	Crack
D20	Crack
D40	Pothole
D43	Ignored
D44	Ignored
D50	Ignored

Images containing both a crack label and D40 are assigned to Both.

Images containing no recognised damage are assigned to Normal.

Automated preprocessing

The dataset is automatically converted from object-level annotations into an image-classification dataset.

RDD2022 Images + XML Annotations
              ↓
       Annotation Parser
              ↓
       Classification Rules
              ↓
 ┌────────┬────────┬─────────┬────────┐
 │ Normal │ Crack  │ Pothole │  Both  │
 └────────┴────────┴─────────┴────────┘

No manual image-by-image classification is required.

⚙️ Project Architecture

The intended end-to-end workflow is:

Government Vehicle
       │
       ▼
    Video
       │
       ▼
 Frame Extraction
       │
       ▼
 Standardized Images
       │
       ▼
   Preprocessing
       │
       ▼
 Custom CNN
       │
       ▼
 Damage Classification
       │
       ▼
 Flagged Inspection Results
       │
       ▼
 Municipal Dashboard

The workflow can be orchestrated using n8n, allowing vehicle footage to be processed automatically.

For the current prototype, the core focus is the image-classification component.

🖥️ Prototype

The project includes a local web interface for testing the trained model.

The interface allows a user to provide a road image and receive a classification from the trained CNN.

Example:

Input Image
     ↓
Preprocessing
     ↓
CNN
     ↓
┌──────────────────────┐
│ Prediction: Pothole  │
│ Confidence: 91.4%    │
└──────────────────────┘

The application is designed to run locally rather than requiring an external AI inference service.

📁 Repository Structure
.
├── app.py                  # Local web application
├── predict.py              # Model loading and inference
├── train.py                # CNN training
├── model.py                # CNN architecture
├── preprocess.py           # Image preprocessing
├── checkpoint.py           # Model checkpoint handling
│
├── checkpoints/
│   └── ...                 # Trained model weights
│
├── dataset/
│   ├── Normal/
│   ├── Crack/
│   ├── Pothole/
│   └── Both/
│
├── results/
│   └── ...                 # Training/evaluation results
│
└── README.md

The exact repository structure may vary as development continues.

🚀 Running the Project
1. Clone the repository
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
2. Install dependencies
pip install -r requirements.txt
3. Prepare the dataset

Place the processed dataset into the appropriate dataset directory.

The expected classes are:

Normal/
Crack/
Pothole/
Both/
4. Train the model
python train.py

Training produces a checkpoint containing the learned model parameters.

5. Run the application
python app.py

The local web application can then be accessed through the address displayed by Flask.

📈 Current Status

This project is currently a hackathon prototype.

Implemented
 RDD2022 Indian dataset preprocessing
 XML annotation parsing
 Automated classification into four categories
 Custom CNN architecture
 Training from scratch
 Model checkpointing
 Local inference
 Local web interface
In development / future work
 Automated video-to-frame pipeline
 n8n integration
 GPS/location metadata
 Municipal inspection dashboard
 Persistent inspection database
 Road-damage severity estimation
 Sidewalk damage classification
 Road-sign deterioration detection
 Graffiti and urban-decay detection
 Cross-country dataset validation
 Edge deployment on municipal hardware
⚠️ Limitations

The current model is a proof of concept and should not be treated as a production-grade road inspection system.

The model has been trained primarily on the Indian RDD2022 dataset. Real-world deployment would require additional validation across:

Different countries
Different road surfaces
Weather conditions
Lighting conditions
Camera systems
Vehicle speeds
Image quality
Seasonal changes

Further data and testing would also be required before using predictions to make actual infrastructure-maintenance decisions.

The intended workflow therefore remains human-in-the-loop:

AI detects potential issue
          ↓
Municipal staff review
          ↓
Maintenance decision

The AI assists inspectors rather than replacing them.

🌍 Future Vision

The current prototype focuses on four road conditions, but the underlying architecture can be extended into a broader municipal infrastructure-monitoring platform.

Future models could identify:

Road Damage
├── Potholes
├── Cracks
└── Other surface deterioration

Sidewalk Damage
├── Broken pavement
├── Uneven surfaces
└── Obstructions

Urban Decay
├── Graffiti
├── Damaged benches
└── Other public infrastructure damage

Road Sign Condition
├── Faded signs
├── Damaged signs
└── Obstructed signs

Combined with GPS information and historical inspections, this could eventually allow municipalities to maintain an evolving map of infrastructure condition.

🏛️ Design Philosophy

This project is built around three principles:

1. Data Sovereignty

Municipal data should remain under municipal control.

2. Small, Explainable Infrastructure

A municipality should not need a massive AI model or an external AI provider to perform a focused computer-vision task.

3. Human-in-the-Loop Automation

AI should reduce repetitive inspection work while leaving final maintenance decisions to qualified municipal staff.

👥 Hackathon Project

Built for NGN Hackathon 2026.

The project explores how locally deployed, purpose-built AI can help municipalities process infrastructure imagery at scale while maintaining control over their data.

Built from scratch. Built for local deployment. Built to assist the people maintaining our roads.
