from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import cv2
import time

# Initialize the feature extractor and model
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
feature_extractor.reduce_labels = False
feature_extractor.size = 128

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=3,
                                                         ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model_apple_v3.pt"))
model.eval()
model.to(device)

# Define the color map
color_map = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}


def prediction_to_vis(prediction):
    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape, dtype=np.uint8)
    for i, c in color_map.items():
        vis[prediction == i] = c
    return Image.fromarray(vis)


# Open the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    start = time.perf_counter()
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (640, 640))
    img_pil = Image.fromarray(img).convert('RGB')
    encoded_inputs = feature_extractor(images=img_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoded_inputs)

    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy().squeeze()
    end = time.perf_counter()

    mask = prediction_to_vis(predicted_mask)
    overlay_img = Image.blend(img_pil, mask, alpha=0.3)
    overlay_img = np.asarray(overlay_img)

    total_time = end - start
    fps = 1 / total_time

    # Ensure overlay_img is writable
    overlay_img = overlay_img.copy()

    # Add FPS text to the image
    cv2.putText(overlay_img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Img', overlay_img)

    if cv2.waitKey(5) == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()

