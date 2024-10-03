import cv2
import mediapipe as mp

# Función para obtener la posición relativa de la mano
def obtener_posicion_mano(hand_landmarks, frame_shape):
    puntos_mano = []
    altura, ancho, _ = frame_shape
    
    for punto, landmark in enumerate(mp.solutions.hands.HandLandmark):
        posicion_landmark = hand_landmarks.landmark[landmark]
        x, y = int(posicion_landmark.x * ancho), int(posicion_landmark.y * altura)
        puntos_mano.append((x, y, punto))  # Agregar el número del landmark
    
    return puntos_mano

# Función para dibujar el texto en la imagen
def dibujar_texto(frame, texto, posicion, color_texto):
    cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2, cv2.LINE_AA)

# Función para dibujar los números de los landmarks
def dibujar_numeros(frame, puntos_mano):
    for (x, y, num) in puntos_mano:
        cv2.putText(frame, str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

# Función para dibujar las conexiones entre los landmarks
def dibujar_conexiones(frame, hand_landmarks, mp_hands):
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]
        start_pos = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
        end_pos = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
        cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Configuración de Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Permitir hasta dos manos

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir frame.")
        break

    # Voltear horizontalmente la imagen
    frame = cv2.flip(frame, 1)

    # Convertir el frame a RGB (Mediapipe requiere RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos en el frame
    resultados = hands.process(frame_rgb)

    # Inicializar variables para la detección de manos
    mano_izquierda_detectada = False
    mano_derecha_detectada = False

    # Extraer puntos clave de las manos detectadas
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            # Obtener los puntos clave de la mano con sus números
            puntos_mano = obtener_posicion_mano(hand_landmarks, frame.shape)
            
            # Dibujar los puntos clave de la mano en el frame
            for (x, y, num) in puntos_mano:
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Dibujar un círculo en el punto

            # Mostrar el número de cada punto clave
            dibujar_numeros(frame, puntos_mano)

            # Dibujar conexiones entre los puntos de la mano
            dibujar_conexiones(frame, hand_landmarks, mp_hands)

            # Determinar la posición relativa de la mano
            if puntos_mano[mp.solutions.hands.HandLandmark.WRIST][0] < puntos_mano[mp.solutions.hands.HandLandmark.THUMB_TIP][0]:
                mano_texto = "Izquierda"
                color_texto_mano = (0, 255, 255)  # Amarillo para mano izquierda
                mano_izquierda_detectada = True
            else:
                mano_texto = "Derecha"
                color_texto_mano = (0, 255, 0)  # Verde para mano derecha
                mano_derecha_detectada = True

            # Mostrar el nombre de la mano (derecha o izquierda)
            if mano_texto == "Izquierda":
                posicion_texto_mano = (10, 30)  # Posición para mano izquierda
            else:
                posicion_texto_mano = (frame.shape[1] - 150, 30)  # Posición para mano derecha

            dibujar_texto(frame, f"{mano_texto}", posicion_texto_mano, color_texto_mano)

    # Mostrar el frame con los puntos clave de la mano
    cv2.imshow('Detección de Mano', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
