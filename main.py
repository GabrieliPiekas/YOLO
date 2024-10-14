import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration
import av
from utils import load_class_names
from ultralytics import YOLO
from PIL import Image

# Carregar a rede YOLOv4 personalizada
net = cv2.dnn.readNet('config_yolo/yolov4_custom_best.weights', 'config_yolo/yolov4_custom.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar as classes personalizadas
class_names = load_class_names("config_yolo/obj.names")

# Carregar o modelo YOLOv8
model = YOLO("config_yolo/best.pt")

def detect_objects_yolov4(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []

    # Dicionário para contar as classes detectadas
    class_counts = {class_name: 0 for class_name in class_names}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_with_confidence = f"{label} {confidence * 100:.2f}%"
            cv2.putText(image, label_with_confidence, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            class_counts[label] += 1

    return image, class_counts

def detect_objects_yolov8(image):
    # Fazer a detecção
    results = model.predict(image)

    # Acessar os nomes das classes a partir do modelo
    class_names = model.names  # Isso acessa os nomes das classes
    class_counts = {class_name: 0 for class_name in class_names.values()}

    # Processar as detecções
    for detection in results:
        for box in detection.boxes:
            class_id = int(box.cls)
            label = class_names[class_id]
            
            # Incrementar a contagem da classe correspondente
            class_counts[label] += 1
            
            # Desenhar a caixa delimitadora e o rótulo
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da caixa
            confidence = box.conf[0]  # Confiança da detecção

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_with_confidence = f"{label} {confidence * 100:.2f}%"
            cv2.putText(image, label_with_confidence, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, class_counts

# Classe do processador de vídeo
class VideoProcessor(VideoProcessorBase):
    def __init__(self, detection_method):
        self.class_counts = {}
        self.detection_method = detection_method

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.detection_method == "YOLOv4":
            result_frame, class_counts = detect_objects_yolov4(img)
        else:
            result_frame, class_counts = detect_objects_yolov8(img)

        self.class_counts = class_counts

        return av.VideoFrame.from_ndarray(result_frame, format="bgr24")

# Função principal para o Streamlit
def main():
    st.title("YOLO Custom Object Detection")

    # Adicionar opção para escolher entre YOLOv4 e YOLOv8
    yolo_version = st.selectbox('Escolha a versão do YOLO para detecção', ('YOLOv4', 'YOLOv8'))

    # Adicionar opção para escolher entre imagem ou câmera ao vivo
    option = st.selectbox('Escolha a entrada para detecção', ('Imagem', 'Câmera ao vivo'))

    if option == 'Imagem':
        uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            if yolo_version == 'YOLOv4':
                result_image, class_counts = detect_objects_yolov4(image)
            else:
                result_image, class_counts = detect_objects_yolov8(image)

            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_image_rgb, channels="RGB", use_column_width=True)
            st.write("Contagem de Classes:")
            st.json(class_counts)

    elif option == 'Câmera ao vivo':
        st.write(f"A detecção ao vivo com {yolo_version} será exibida abaixo.")

        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun2.l.google.com:19302"]}]})

        webrtc_ctx = webrtc_streamer(
            key="detector",
            video_processor_factory=lambda: VideoProcessor(yolo_version),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        class_counts_placeholder = st.empty()
        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                class_counts = webrtc_ctx.video_processor.class_counts
                with class_counts_placeholder.container():
                    st.write("Contagem de Classes Detectadas:")
                    st.json(class_counts)

if __name__ == '__main__':
    main()
