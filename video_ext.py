# Import libraries
import cv2
import re
import os 
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

## Initial setup

VIDEO_DIR = "test_plates"
INPUT_VIDEO_FILE = os.path.join(VIDEO_DIR, "plat.mp4")
OUTPUT_VIDEO_FILE = os.path.join(VIDEO_DIR, "output_plate_detection.mp4")

# Create the folder if it does not exists
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
    print(f"--- Directorio creado: {VIDEO_DIR} ---")
# ----------------------------------------

#Defined patterns for the license plates
PATTERN_LLLNNNL = re.compile(r'^([A-Z]{3})([0-9]{3})([A-Z]{1})$') 
PATTERN_LLLNNNN = re.compile(r'^([A-Z]{3})([0-9]{4})$') 
PATTERN_LNNLLL = re.compile(r'^([A-Z]{1})([0-9]{2})([A-Z]{3})$')


# Variable created to control the frame detection
FRAME_SKIP = 6
frame_count = 0 

# Initialize the YOLO model and PaddleOCR
print("--- Inicializando modelos: YOLO y PaddleOCR ---")
model = YOLO("best.pt")
ocr = PaddleOCR(use_textline_orientation=False, lang='es') 

#Video settings
cap = cv2.VideoCapture(INPUT_VIDEO_FILE)

if not cap.isOpened():
    print(f"ERROR: La fuente de video no se puede abrir: {INPUT_VIDEO_FILE}")
    print("Asegúrate de que el archivo 'plate.mp4' exista dentro de la carpeta 'test_plates'.")
    exit()

# Define the input video resolution
real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) 

print(f"--- Resolución del video de entrada: {real_width}x{real_height} ({fps:.2f} FPS) ---")

# Output video configuration (codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, fps, (real_width, real_height))
if not video_writer.isOpened():
    print(f"ERROR: No se pudo inicializar VideoWriter para guardar en {OUTPUT_VIDEO_FILE}. Intente cambiar el códec (e.g., a 'DIVX').")
    cap.release()
    exit()
# -----------------------------------------------------

## Functions

def preprocess_for_ocr(img_crop): # Get the cut license image
   # ROI masking
    h, w, _ = img_crop.shape

    #Create a mask to focus on the central area of the license plate
    x_start = int(w * 0.05)   # 5% horizontal margin
    x_end = int(w * 0.95)     # 5% horizontal margin
    y_start = int(h * 0.05)   # 5% superior vertical margin
    y_end = int(h * 0.90)     # 65% inferior vertical margin

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
    
    # 6 characters format
    elif len(base_plate) == 6:
        # Pattern LNNLLL (1 Letter, 2 Numbers, 3 Letters)
        if PATTERN_LNNLLL.fullmatch(base_plate):
            L = base_plate[0]
            NN = base_plate[1:3]
            LLL = base_plate[3:6]
            output_text = f"{L}{NN}-{LLL}" # LNN-LLL

    return output_text


# Variable to storage the last license detecion
last_plate_info = {"text": "", "x1": 0, "y1": 0, "x2": 0, "y2": 0}
plate_found_count = 0

print("--- Procesando video, esto puede tardar... ---")

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

                    #Execute OCR con manejo de excepción para el 'Error al procesar OCR: 1'
                    try:
                        result_ocr = ocr.predict(preprocessed_plate_3channel) 
                        
                        if result_ocr and result_ocr[0]: 
                            data = result_ocr[0] 
                            
                            # Adapt the extraction with Paddle format
                            if isinstance(data, list):
                                text_list = [item[1][0] for item in data]
                            elif isinstance(data, dict) and 'rec_texts' in data:
                                text_list = data['rec_texts']
                            else:
                                text_list = []
                            
                            raw_text_combined = ''.join(text_list)
                            output_text = format_plate_text(raw_text_combined)
                            
                            if output_text:
                                plate_found_count += 1
                                print(f"Frame {frame_count}: Matrícula Encontrada: {output_text} (VÁLIDO)")
                                
                                #Save the last valid plate info
                                last_plate_info.update({"text": output_text, 
                                                         "x1": current_x1, "y1": current_y1, 
                                                         "x2": current_x2, "y2": current_y2})
                                found_plate_in_frame = True
                                break 
                            # else:
                                 # print(f"DEBUG: Fund Text ({raw_text_combined}), but doees no match with a pattern.")
                        
                    except Exception as e:
                        # print(f"Error while processing OCR in frame {frame_count}: {e}")
                        pass
            
            if found_plate_in_frame:
                break 

        #License plate detected but no valid text found
        #Cleaning last plate info if no plate found in current frame
        if temp_box and not found_plate_in_frame:
            last_plate_info["text"] = "" 
    
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

    # --- Save the frame in the output video. ---
    video_writer.write(frame)

#Resource cleanup
cap.release()
video_writer.release()
cv2.destroyAllWindows() 

print(f"\n--- Proceso finalizado ---")
print(f"Frames totales procesados: {frame_count}")
print(f"Matrículas válidas encontradas: {plate_found_count}")
print(f"El video con las detecciones se ha guardado en: {OUTPUT_VIDEO_FILE}")