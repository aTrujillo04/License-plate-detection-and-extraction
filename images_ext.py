# Import libraries 
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import imutils
import re
import numpy as np # Added numpy for preprocessing

# Define the path for the image you want to test
image = cv2.imread("./test_plates/test1.jpeg")

# Get original image dimensions
orig_h, orig_w, _ = image.shape

# Initialize YOLO model and PaddleOCR
model = YOLO("best.pt") # Trained YOLO model for license plate detection
ocr = PaddleOCR(use_angle_cls=True, lang='en') # OCR with English language support and angle classification

#Pattern to extract from the license plate

PATTERN_LLLNNNL = re.compile(r'([A-Z]{3})([0-9]{3})([A-Z]{1})') # 7 chars
PATTERN_LLLNNNN = re.compile(r'([A-Z]{3})([0-9]{4})')          # 7 chars
PATTERN_LNNLLL = re.compile(r'([A-Z]{1})([0-9]{2})([A-Z]{3})')  # 6 chars


def preprocess_for_ocr(img_crop): # Receives the cropped license plate image
   # Preprocessing function with Otsu thresholding (SIN MÁSCARA)
   
    # Convert to Grayscale and apply Median Blur
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) 
    gray = cv2.medianBlur(gray, 3) 
    
    # Compute Otsu's thresholding and binarize the image (black text on white background)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Morphological operations to fill little holes and reduce noise
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return thresh # Return the 1-channel binary image

def format_plate_text(text): #Function that receives the raw text
    # Clean and format the OCR text to match known license plate patterns.
    # Clean the text: uppercase, remove spaces and non-alphanumeric characters
    raw_text = text.upper().replace(" ", "")
    raw_text = re.sub(r'[^A-Z0-9]', '', raw_text)
   
    #It will look for patterns inside the raw text
    
    match = PATTERN_LLLNNNL.search(raw_text) #Look for the pattern
    if match:
        #Match realized
        LLL = match.group(1) # YRA
        NNN = match.group(2) # 049
        L = match.group(3)   # B
        return f"{LLL}-{NNN}-{L}" 

    match = PATTERN_LLLNNNN.search(raw_text) #Look for the pattern
    if match:
        LLL = match.group(1)
        NNNN = match.group(2)
        N1 = NNNN[0:2]
        N2 = NNNN[2:4]
        return f"{LLL}-{N1}-{N2}" # LLL-NN-NN
    
    match = PATTERN_LNNLLL.search(raw_text) #Look for the pattern
    if match:
        L = match.group(1)
        NN = match.group(2)
        LLL = match.group(3)
        return f"{L}{NN}-{LLL}" # LNN-LLL

    #Return if no pattern is found
    return ""

# Execute YOLO model on the image printing the box
results = model(image)
print(results[0].boxes)

for result in results:
    # Show detected boxes for class 0 (license plates)
    index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
    print(index_plates)

    for idx in index_plates:
        # Get confidence score
        conf = result.boxes.conf[idx].item()
        if conf > 0.3:
            # Obtain bounding box coordinates
            xyxy = result.boxes.xyxy[idx].squeeze().tolist()
            x1, y1 = int(xyxy[0]), int(xyxy[1])
            x2, y2 = int(xyxy[2]), int(xyxy[3])
            
            # Cut the license plate from the image with some padding
            padding = 15
            y1_pad = max(0, y1 - padding)
            y2_pad = min(orig_h, y2 + padding)
            x1_pad = max(0, x1 - padding)
            x2_pad = min(orig_w, x2 + padding)
            
            plate_image = image[y1_pad:y2_pad, x1_pad:x2_pad]

            if plate_image.size == 0:
                continue

            # 1. Preprocess the crop (ahora sin máscara)
            preprocessed_plate = preprocess_for_ocr(plate_image.copy())
            
            # 2. Convert binary image back to 3-channel for Paddle
            preprocessed_plate_3channel = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR)

            # 3. Execute OCR on the *preprocessed* image
            result_ocr = ocr.predict(preprocessed_plate_3channel)
            print(result_ocr)
            
            output_text = "" # Initialize empty text

            #Look for not empty detections 
            if result_ocr and result_ocr[0]:
                try:
                    boxes = result_ocr[0]['rec_boxes']
                    texts = result_ocr[0]['rec_texts']
                    left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
                    print(f"left_to_right:", left_to_right)
                    
                    # Combine text
                    raw_text_combined = ''.join([t for _, t in left_to_right])
                    
                    # Format text
                    output_text = format_plate_text(raw_text_combined)
                    print(f"Raw: {raw_text_combined} -> Formatted: {output_text}")

                except Exception as e:
                    print(f"Error processing OCR: {e}")
                    pass
            
            # Visualize the cropped plate images
            cv2.imshow("Original Cropped Plate", plate_image)
            cv2.imshow("Preprocessed B&W Plate (No Mask)", preprocessed_plate)
            
            # Draw bounding box and recognized text on the original image
            if output_text: 
                box_color = (0, 255, 0) # Green
                cv2.rectangle(image, (x1 - 10, y1 - 35), (x2 + 10, y2-(y2 -y1)), box_color, -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(image, output_text, (x1-7, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            else: # If no valid text was found
                box_color = (0, 0, 255) # Red
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            
# Show final image with detections and OCR results
cv2.imshow("Image", imutils.resize(image, width=720))
cv2.waitKey(0)
cv2.destroyAllWindows()