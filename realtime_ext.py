import cv2
import re
import os # NUEVO: Para guardar imágenes
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- 1. Configuración Inicial ---
model = YOLO("  best.pt")
ocr = PaddleOCR(use_textline_orientation=False, lang='es')
whitelist_pattern = re.compile(r'^[A-Z0-9]+$')

# --- 2. Configuración de la Cámara ---
video_source = 2
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la fuente de video: {video_source}")
    exit()

# --- NUEVO: PASO 1 (Verificar Resulución Real) ---
# Intentamos poner 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Ahora leemos la resolución que la cámara REALMENTE aceptó
real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"--- Resolución real de la cámara: {real_width}x{real_height} ---")
# ----------------------------------------------------

# (Tu función de pre-procesamiento va aquí)
def preprocess_for_ocr(img_crop):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Intenta también invirtiendo. A veces el OCR prefiere texto negro
    # thresh = 255 - thresh 
    
    return thresh

print("Iniciando... Presiona 'q' para salir. Presiona 's' para GUARDAR un recorte.")

# --- 3. Bucle Principal ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = model(frame)

    for result in results:
        index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]

        for idx in index_plates:
            conf = result.boxes.conf[idx].item()
            
            if conf > 0.7:
                xyxy = result.boxes.xyxy[idx].squeeze().tolist()
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                
                y1_pad = max(0, y1 - 15)
                y2_pad = min(h, y2 + 15)
                x1_pad = max(0, x1 - 15)
                x2_pad = min(w, x2 + 15)
                
                plate_image = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                if plate_image.size == 0:
                    continue

                # --- NUEVO: PASO 2 (Verificar Tamaño del Recorte) ---
                # ----------------------------------------------------

                preprocessed_plate = preprocess_for_ocr(plate_image)
                preprocessed_plate_3channel = cv2.cvtColor(preprocessed_plate, cv2.COLOR_GRAY2BGR)

                # Pasamos la imagen pre-procesada
                result_ocr = ocr.predict(preprocessed_plate_3channel) 
                print(f"RAW OCR: {result_ocr}")

                output_text = ""
                
                if result_ocr and result_ocr[0]:
                    try:
                        data = result_ocr[0]
                        text_list = data['rec_texts']
                        
                        # Simplificado para el ejemplo:
                        raw_text = ''.join(text_list) 
                        output_text = ''.join([char for char in raw_text if whitelist_pattern.fullmatch(char)])
                        print(f"RAW OCR Text: {raw_text} -> Filtrado: {output_text}")
                        
                    except Exception as e:
                        print(f"Error procesando OCR: {e}")
                        pass
                
                # Mostrar la imagen pre-procesada (la B/N)
                cv2.imshow("Placa Recortada (B/N para OCR)", preprocessed_plate)
                
                # Dibujar resultados
                cv2.rectangle(frame, (x1, y1-30), (x1+180, y1), (0, 255, 0), -1)
                cv2.putText(frame, output_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0 , 0), 2)


    cv2.imshow("Deteccion y OCR en vivo", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    # --- NUEVO: PASO 3 (Guardar imágenes de debug) ---
    if key == ord('s'):
        if 'plate_image' in locals() and plate_image.size > 0:
            cv2.imwrite("debug_recorte_color.png", plate_image)
            cv2.imwrite("debug_recorte_bn.png", preprocessed_plate)
            print("--- ¡Imágenes de debug guardadas! (color y b/n) ---")
        else:
            print("--- No hay ninguna placa detectada para guardar ---")
    # ----------------------------------------------------

cap.release()
cv2.destroyAllWindows()