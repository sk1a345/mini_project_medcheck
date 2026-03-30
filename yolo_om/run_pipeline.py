from yolo_inference import run_yolo
from mapping import map_yolo_output
from decision_test import second_scan

def run(pack_id, image_path):
    detections = run_yolo(image_path)
    # Perception Layer
    # Takes an image of the blister pack
    # Runs YOLOv8 model
    # Outputs raw detections (bounding boxes, class, confidence)

    yolo_output = map_yolo_output(detections)
    # “Organize what I see into meaningful slots”
        # Output might look like:
        # {
        #   "cavity_1": "intact",
        #   "cavity_2": "broken",
        #   "cavity_3": "intact"
        # }
    second_scan(pack_id, yolo_output, trigger="motion")
        #     👉 Agent / Decision Layer
        
        # This is where your system becomes intelligent.
        
        # Inputs:
        
        # pack_id → identifies user/medicine pack
        # yolo_output → current state
        # trigger="motion" → event-based activation
        
        # This function likely:
        
        # Compares current state with past state
        # Detects:
        # pill taken
        # repeated attempt (danger)
        # Uses memory + risk scoring
        # Decides whether to trigger alert
        # Think of it as:
        
        # “Is this action safe or dangerous?”
if __name__ == "__main__":
    pack_id = input("Enter pack_id: ")
    image_path = input("Enter image path: ")

    run(pack_id, image_path)
