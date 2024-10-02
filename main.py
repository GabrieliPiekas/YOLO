import streamlit as st
import cv2
import numpy as np
import os
import gdown

from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from utils import load_class_names


# Função para baixar o arquivo de pesos do Google Drive
def download_weights_from_drive(drive_file_id, destination):
    # URL do arquivo no Google Drive
    drive_url = f'https://drive.google.com/uc?id={drive_file_id}'
    
    # Verificar se o arquivo já foi baixado
    if not os.path.exists(destination):
        print("Baixando os pesos do Google Drive...")
        gdown.download(drive_url, destination, quiet=False)
        print("Download concluído.")
    else:
        print("Pesos já estão disponíveis localmente.")
 
# ID do arquivo no Google Drive
file_id = '1-1dUAZJB7yji54y3R9BCL0_DafYKfQTr'  # Substitua pelo seu ID real
destination = 'config_yolo/yolov4_custom_best.weights'
        
# URL do arquivo .cfg no Google Drive
cfg_url = "https://drive.google.com/18uCxw2Z7fajZ1e2ztOrDTvsM9s4OpDN-O"

# Caminho onde o arquivo será salvo
cfg_path = "config_yolo/yolov4_custom.cfg"

# Verifica se o arquivo já existe antes de baixar
if not os.path.exists(cfg_path):
    gdown.download(cfg_url, cfg_path, quiet=False)


# Baixar os pesos do Google Drive
download_weights_from_drive(file_id, destination)

# Carregar a rede YOLOv4 personalizada
net = cv2.dnn.readNet(destination, cfg_url)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar as classes personalizadas
class_names = load_class_names("config_yolo/obj.names")

# Função para processar a imagem
def detect_objects(image):
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
            if confidence > 0.7:  # Configurar precisão
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
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]
            
            # Desenhar o retângulo da caixa delimitadora
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Texto do rótulo com a classe e a confiança (precisão) em porcentagem
            label_with_confidence = f"{label} {confidence * 100:.2f}%"
            
            # Adicionar o texto da precisão
            cv2.putText(image, label_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Incrementar a contagem da classe correspondente
            class_counts[label] += 1

    return image, class_counts

# Classe do processador de vídeo
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.class_counts = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Fazer a detecção de objetos
        result_frame, class_counts = detect_objects(img)
        self.class_counts = class_counts
        
        return av.VideoFrame.from_ndarray(result_frame, format="bgr24")

# Função principal para o Streamlit
def main():
    st.title("YOLOv4 Custom Object Detection")

    # Adicionar opção para escolher entre imagem ou câmera ao vivo
    option = st.selectbox('Escolha a entrada para detecção', ('Imagem', 'Câmera ao vivo'))

    if option == 'Imagem':    
        # Carregar a imagem
        uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Detecção de objetos
            result_image, class_counts = detect_objects(image)

            # Converter de BGR para RGB
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            # Exibir a imagem com as detecções
            st.image(result_image_rgb, channels="RGB", use_column_width=True)
            
            # Exibir contagem das classes
            st.write("Contagem de Classes:")
            st.json(class_counts)  # Exibir as contagens em formato JSON

    elif option == 'Câmera ao vivo':
        st.write("A detecção ao vivo será exibida abaixo.")
        webrtc_ctx = webrtc_streamer(
            key="detector",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            st.write("Contagem de Classes Detectadas:", webrtc_ctx.video_processor.class_counts)

if __name__ == '__main__':
    main()
