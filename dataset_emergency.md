Emergency vehicle detection — dataset & training guidance

Goal

- Add reliable detection of emergency vehicles (ambulance, firetruck) for the traffic management system.

Approaches

1) Fine-tune YOLOv8 to include ambulance/firetruck classes (recommended for production-quality results).
2) Use a lightweight heuristic on top of a general vehicle detector to detect emergency roof lights (fast, works reasonably well where lights are visible).

Recommended dataset structure (YOLOv8 format)

- dataset/
  - images/
    - train/
    - val/
  - labels/
    - train/
    - val/

- Classes: 0=car, 1=truck, 2=motorcycle, 3=bus, 4=bicycle, 5=ambulance, 6=firetruck

Labeling

- Use LabelImg, Roboflow, CVAT, or makesense.ai and export to YOLO format.
- Make sure to label both the vehicle bounding box and the class correctly.
- For ambulances and firetrucks, include images with visible roof lights when possible.

Data collection tips

- Collect images from many angles and lighting conditions.
- Include cropped close-ups and wide shots (vehicle may be small in frame).
- Include images with occlusion, different colors, and parked/moving vehicles.
- Augment: random brightness/contrast, horizontal flips, small rotations, cutout, mosaic.

Training with Ultralytics YOLOv8

- Install ultralytics: pip install -U ultralytics
- Create a YAML config file (e.g. emergency_dataset.yaml):

  path: ./dataset
  train: images/train
  val: images/val
  nc: 7
  names: ['car','truck','motorcycle','bus','bicycle','ambulance','firetruck']

- Train command (PowerShell):

  python -m ultralytics train model=yolov8s.pt data=emergency_dataset.yaml epochs=50 imgsz=640 batch=16

- Monitor results and tune learning rate / epochs / augmentations.

Using the trained model

- Copy the best .pt into the project and set env var YOLO_EMERGENCY_MODEL or update `config.EMERGENCY_MODEL` in `app.py`.
- Set `config.USE_EMERGENCY_HEURISTIC = False` when you want only model-based detection.

Quick heuristic (what's implemented in code)

- If the detector finds a car/truck/bus and the model doesn't report ambulance/firetruck, the app crops the roof area and checks for bright red/blue pixels (siren lights).
- Heuristic is fast and reduces false negatives in many scenes but cannot replace a trained model in all conditions.

Evaluation

- Create a small validation split and compute mAP for emergency classes after training.
- For heuristic, create a test set and compute precision/recall (count TP/FP/FN manually or with scripts).

Resources

- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/
- Roboflow: https://roboflow.com/ (dataset management and augmentation)
- LabelImg: https://github.com/tzutalin/labelImg

Contact

If you want, I can provide a small helper script to convert annotations or a starter training YAML — tell me which step you'd like next.
