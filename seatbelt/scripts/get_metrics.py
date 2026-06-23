from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\stage4\weights\best.pt")
    metrics = model.val(data=r"c:\Users\Acer\OneDrive\Desktop\seatbelt\outputs\cropped_cls_dataset", workers=0)
    cm = metrics.confusion_matrix.matrix

    print("NAMES:", model.names)
    print("CONFUSION MATRIX RAW:")
    print(cm)
