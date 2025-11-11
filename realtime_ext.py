# Import libraries
import cv2
import re
import os 
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

#Initial setup

#Here, we define the patterns for different license plate formats.
PATTERN_LLLNNNL = re.compile(r'^([A-Z]{3})([0-9]{3})([A-Z]{1})$') 
PATTERN_LLLNNNN = re.compile(r'^([A-Z]{3})([0-9]{4})$') 
PATTERN_LNNLLL = re.compile(r'^([A-Z]{1})([0-9]{2})([A-Z]{3})$')


# Variable to control frame skipping and control
FRAME_SKIP = 5 
frame_count = 0 

# Initialize YOLO trained model and PaddleOCR
model = YOLO("best.pt")
ocr = PaddleOCR(use_textline_orientation=False, lang='es') 

# Camera setup
video_source = 2
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Video source cannot be opened: {video_source}")
    exit()

# Defined resolution setup
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Read and print real camera resolution
real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"--- Real camera resolution: {real_width}x{real_height} ---")

#Functions

def preprocess_for_ocr(img_crop): # Receives the cropped license plate image
   #Preprocessing function with ROI masking and Otsu thresholding
    h, w, _ = img_crop.shape

    #Create a mask to focus on the central area of the license plate
    x_start = int(w * 0.05)   # 5% horizontal margin
    x_end = int(w * 0.95)     # 5% horizontal margin
    y_start = int(h * 0.05)   # 5% superior vertical margin
    y_end = int(h * 0.65)     # 65% inferior vertical margin

    mask = np.zeros(img_crop.shape, dtype=np.uint8)
    cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)
    
    #Apply the mask to the cropped image
    masked_img = cv2.bitwise_and(img_crop, mask)

    #Convert to Grayscale and apply Median Blur
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.medianBlur(gray, 3) 
    
    #Compute Otsu's thresholding and binarize the image (black text on white background)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #Morphological operations to clean up noise
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return thresh 

def format_plate_text(text):
    #Clean and format the OCR text to match known license plate patterns.
    #Clean the text: uppercase, remove spaces and non-alphanumeric characters
    raw_text = text.upper().replace(" ", "")
    raw_text = re.sub(r'[^A-Z0-9]', '', raw_text)

    #Extract the base plate sequence
    #Look for the sequence
    # The pattern [A-Z0-9]{6,8} looks for sequences of 6 to 8 alphanumeric characters
    match = re.search(r'[A-Z0-9]{6,8}', raw_text)
    
    if not match:
        return ""

    base_plate = match.group(0) #Get the sequence (e.g., "NZY2511", "MNR952A", "A12BCD")
    
    #Verify against known patterns and format accordingly
    output_text = ""
    
    #7 characters format
    if len(base_plate) == 7:
        #First attempt: Pattern LLLNNNN (3 Letters, 4 Numbers)
        if PATTERN_LLLNNNN.fullmatch(base_plate):
            LLL = base_plate[0:3]
            N1 = base_plate[3:5]
            N2 = base_plate[5:7]
            output_text = f"{LLL}-{N1}-{N2}" # LLL-NN-NN
        
        # Second attempt: Pattern LLLNNNL (3 Letters, 3 Numbers, 1 Letter)
        elif PATTERN_LLLNNNL.fullmatch(base_plate):
            LLL = base_plate[0:3]
            NNN = base_plate[3:6]
            L = base_plate[6]
            output_text = f"{LLL}-{NNN}-{L}" # LLL-NNN-L
    
    # 8 characters format
    elif len(base_plate) == 6:
        # Pattern LNNLLL (1 Letter, 2 Numbers, 3 Letters)
        if PATTERN_LNNLLL.fullmatch(base_plate):
            L = base_plate[0]
            NN = base_plate[1:3]
            LLL = base_plate[3:6]
            output_text = f"{L}{NN}-{LLL}" # LNN-LLL

    #Add more possible formats here

    return output_text


# Instructions for the user 
print("Initializing, press --q-- to exit and --s-- to save the image.")

# --- Bucle Principal ---

# Variables to store the last successfully read plate information
last_plate_info = {"text": "", "x1": 0, "y1": 0, "x2": 0, "y2": 0}

while True:
    ret, frame = cap.read() #Next frame
    if not ret:
        break

    frame_count += 1
    
    #Processing control: process only every FRAME_SKIP frames
    process_frame = (frame_count % FRAME_SKIP == 0)

    h, w, _ = frame.shape 
    
    #Initialize variables for this frame
    plate_image = None
    preprocessed_plate = None
    output_text = "" 
    current_x1, current_y1, current_x2, current_y2 = 0, 0, 0, 0
    temp_box = None #Store the actual box

    #Detection and OCR processing
    if process_frame:
        results = model(frame, verbose=False) 
        found_plate_in_frame = False

        for result in results:
            index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0] 

            for idx in index_plates:
                conf = result.boxes.conf[idx].item() 
                
                if conf > 0.7: 
                    xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                    current_x1, current_y1 = int(xyxy[0]), int(xyxy[1])
                    current_x2, current_y2 = int(xyxy[2]), int(xyxy[3])
                    temp_box = (current_x1, current_y1, current_x2, current_y2) #Save the box
                    
                    padding = 25 
                    y1_pad = max(0, current_y1 - padding)
                    y2_pad = min(h, current_y2 + padding)
                    x1_pad = max(0, current_x1 - padding)
                    x2_pad = min(w, current_x2 + padding) 
                    
                    plate_image = frame[y1_pad:y2_pad, x1_pad:x2_pad] 

                    if plate_image.size == 0: 
                        continue

                    #Preprocessing for OCR
                    preprocessed_plate = preprocess_for_ocr(plate_image) 
                    preprocessed_plate_3channel = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR) 

                    #Execute OCR
                    result_ocr = ocr.predict(preprocessed_plate_3channel) 
                    
                    if result_ocr and result_ocr[0]: 
                        try:
                            data = result_ocr[0] 
                            text_list = data['rec_texts']
                            
                            #Logic to combine multiple text segments if needed
                            raw_text_combined = ''.join(text_list)
                            output_text = format_plate_text(raw_text_combined)
                            
                            if output_text:
                                print(f"Matrícula Encontrada: {output_text} (Formato válido)")
                                
                                #Save the last valid plate info
                                last_plate_info.update({"text": output_text, 
                                                         "x1": current_x1, "y1": current_y1, 
                                                         "x2": current_x2, "y2": current_y2})
                                found_plate_in_frame = True
                                break 
                            else:
                                 print(f"DEBUG: Texto encontrado ({raw_text_combined}), pero no coincide con los patrones de formato.")
                                 pass
                        
                        except Exception as e:
                            print(f"Error while processing OCR: {e}")
                            pass
            
            if found_plate_in_frame:
                break 

        #License plate detected but no valid text found
        #Cleaning last plate info if no plate found in current frame
        if temp_box and not found_plate_in_frame:
            last_plate_info["text"] = "" 

    #Show detections and results on the frame
    
    #If valid text was found (either in this frame or from last valid read)
    if output_text or last_plate_info["text"]:
        
        #Using last valid plate info if no new text found
        final_text = output_text if output_text else last_plate_info["text"]
        x1_draw, y1_draw = last_plate_info["x1"], last_plate_info["y1"]
        x2_draw, y2_draw = last_plate_info["x2"], last_plate_info["y2"]

        #Draw the text background and text
        cv2.rectangle(frame, (x1_draw, y1_draw-30), (x1_draw + 250, y1_draw), (0, 150, 0), -1) 
        cv2.putText(frame, final_text, (x1_draw, y1_draw-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255 , 255), 2)
        #Detection box (green)
        cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 150, 0), 2)
        
    #If no valid text found in this frame, but detection box exists
    elif process_frame and temp_box:
        #Detection box (red)
        cv2.rectangle(frame, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), (0, 0, 255), 2)


    #Show cropped images for debugging
    if preprocessed_plate is not None and preprocessed_plate.size > 0:
         #Change the window name
         cv2.imshow("Cut license plate (Masked B/W Processed for OCR)", preprocessed_plate)

    #Show the main frame with detections
    cv2.imshow("OCR and detection in real time", frame)
      
    # Key handling
    key = cv2.waitKey(1) & 0xFF
    
    # Exit on 'q' key
    if key == ord('q'):
        break
    
    # Save cropped images on 's' key
    if key == ord('s'):
        if plate_image is not None and plate_image.size > 0:
            cv2.imwrite("debug_cut_color.png", plate_image)
            cv2.imwrite("debug_cut_bw_processed.png", preprocessed_plate) 
            print("--- Las imágenes (color y B/N preprocesada) fueron guardadas ---")
        else:
            print("--- No license plate to save ---")

#Resource cleanup
cap.release()
cv2.destroyAllWindows()