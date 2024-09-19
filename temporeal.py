import streamlit as st
import cv2
import numpy as np

# Carregar a rede YOLOv4 personalizada
net = cv2.dnn.readNet('YOLO/yolov4_custom_last.weights', 'YOLO/yolov4_custom.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar as classes personalizadas
with open('YOLO/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Função para processar cada frame
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8: # ajuste da precisão
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label + f" {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

st.title("YOLOv4 Real-time Object Detection")

# Capturar vídeo da câmera
run_camera = st.checkbox('Iniciar câmera')

if run_camera:
    # Inicializar captura de vídeo
    cap = cv2.VideoCapture(0)

    # Checar se a câmera está aberta
    if not cap.isOpened():
        st.error("Erro ao abrir a câmera")
    else:
        frame_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Falha ao capturar o frame da câmera.")
                break

            # Redimensionar o frame para se ajustar à tela
            frame = cv2.resize(frame, (640, 480))

            # Fazer detecção de objetos
            result_frame = detect_objects(frame)

            # Exibir o frame processado
            frame_placeholder.image(result_frame, channels="BGR")

        cap.release()
