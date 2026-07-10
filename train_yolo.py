from ultralytics import YOLO

def print_losses(trainer):
    epoch = trainer.epoch + 1
    # Only print if within the training epochs
    if epoch <= trainer.epochs:
        try:
            # Training loss sum (box + cls + dfl)
            train_loss = float(trainer.tloss.sum()) if hasattr(trainer, 'tloss') else 0.0
            
            # Validation loss sum
            metrics = trainer.metrics
            val_loss = metrics.get('val/box_loss', 0.0) + metrics.get('val/cls_loss', 0.0) + metrics.get('val/dfl_loss', 0.0)
            
            print(f"\n--- EPOCH {epoch}/{trainer.epochs} ---")
            print(f"Training Loss:   {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"----------------------\n")
        except Exception:
            pass

def main():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")
    
    # Attach callback to print losses at the end of each epoch
    model.add_callback("on_fit_epoch_end", print_losses)

    # Train the model on the yolo_dataset
    # Adjust epochs and imgsz as needed
    results = model.train(
        data="yolo_dataset/data.yaml",
        epochs=10,
        imgsz=640,
        name="yolov8n_tree_detection",
        workers=0,
        save=False
    )
    
    # Manually save the model at the end to bypass Ultralytics continuous saving lock issues
    model.save("tree_detector_final.pt")
    print("Training complete! Model saved to tree_detector_final.pt")

if __name__ == "__main__":
    main()
