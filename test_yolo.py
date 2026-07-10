from ultralytics import YOLO

def main():
    print("Loading trained model...")
    # Load the model we just trained
    model = YOLO(r"runs/detect/yolov8n_tree_detection-6/weights/best.pt")

    print("\n--- Evaluating Model Metrics on Test Set ---")
    # Evaluate model on the test split defined in data.yaml
    metrics = model.val(data="yolo_dataset/data.yaml", split="test")
    
    print("\n--- Running Inference on Test Images ---")
    # Run predictions on all images in the test folder and save the results
    # The saved images will have bounding boxes drawn on them
    results = model.predict(
        source="yolo_dataset/images/test", 
        save=True,      # Save images with boxes
        conf=0.05,      # Lowered confidence threshold to catch weaker predictions
        name="test_predictions" # Output folder name inside runs/detect/
    )
    
    print("\nTesting complete!")
    print("Metrics have been printed above.")
    print("Visual predictions with bounding boxes are saved in: runs/detect/test_predictions")

if __name__ == "__main__":
    main()
#box_loss:-How wrong the predicted box location is.
#cls_loss:-How wrong the predicted class is. In your case, class is just Tree.
#dfl_loss:-Another box-quality loss YOLO uses to improve box placement.

#mAP50 = 0.3398
#At the loose matching rule IoU >= 0.50, your model scores about 33.98%.
#In simple words: if we are a bit forgiving about box quality, the detector is finding some trees correctly, but performance is still not strong.
#mAP50-95 = 0.10484
#Across strict matching rules from IoU 0.50 up to 0.95, your model averages about 10.48%.
#In simple words: when we demand tighter, more exact boxes, performance drops a lot.

#For each test image, YOLO gives:
#a bounding box
#a class name, here Tree
#a confidence score like 0.92, 0.76, etc.


#For each predicted box, calculate IoU with the real box(if IoU>=0.5 then it  is a correct box match)

#All predictions are sorted from highest confidence to lowest.
#Example:
#0.98
#0.91
#0.83
#0.60
#This matters because high-confidence predictions should ideally be correct first.
#6. Build precision and recall step by step
#As we move down that sorted list, after each prediction we compute:
#Precision = TP / (TP + FP)
#Recall = TP / (TP + FN)
#So after each added prediction, precision and recall change.


#NMS is the cleanup step that removes repeated overlapping boxes and keeps the strongest one for each tree.