import streamlit as st
import cv2
import numpy as np
from utils import load_class_names 

# Carregar a rede YOLOv4 personalizada
net = cv2.dnn.readNet('config_yolo/yolov4_custom_last.weights', 'config_yolo/yolov4_custom.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar as classes personalizadas
with open('config_yolo/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
    
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
            if confidence > 0.7: #configurar precisão
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                # Incrementar a contagem da classe detectada
                # class_counts[class_names[class_id]] += 1
                
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # for i in indices:
    #     i = i[0]
    #     box = boxes[i]
    #     x, y, w, h = box[0], box[1], box[2], box[3]
    #     label = str(classes[class_ids[i]])
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Aplicar Non-Maximum Suppression (NMS) para eliminar sobreposições
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # Verificar se há detecções após o NMS
    if len(indices) > 0:
        indices = indices.flatten() # Usar flatten() se o NMS retornou alguma caixa
        for i in indices.flatten():
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

# Função para iniciar a câmera e fazer a detecção em tempo real
def detect_from_camera():
    cap = cv2.VideoCapture(0)  # Abre a câmera do notebook

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível acessar a câmera.")
            break

        # Fazer a detecção de objetos no frame capturado
        result_frame, class_counts = detect_objects(frame)

        # Exibir o frame com as detecções
        cv2.imshow('Detecção em Tempo Real', result_frame)

        # Fechar com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




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
            st.image(result_image, channels="BGR", use_column_width=True)
            
            # Exibir contagem das classes
            st.write("Contagem de Classes:")
            st.json(class_counts)  # Exibir as contagens em formato JSON
    
    elif option == 'Câmera ao vivo':
        st.write("A detecção ao vivo será iniciada em uma nova janela. Pressione 'q' para sair.")
        if st.button('Iniciar Detecção ao Vivo'):
            detect_from_camera()

if __name__ == '__main__':
    main()
