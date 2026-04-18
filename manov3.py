import cv2 
import pygame as pg
import mediapipe as mp
import webbrowser

#accede al modulo de detección de manos
mp_hands = mp.solutions.hands
#es para dibujar los puntos y conecciones de las manos
mp_draw = mp.solutions.drawing_utils

#variables que se usarán mas tarde
WIDTH, HEIGTH = 800, 600
running = True
pygame_ventana_abierta = False
screen = None
url = "https://youtu.be/GbrGs_2B64U?si=ZWCziR4m3wqCvEE-"
bandera = True

#que cámara usa
cap = cv2.VideoCapture(0)

#se usa para asegurarse de que lo que se está detectando es una mano, para seguir una detectada y para decir cuantas manos puede detectar como máximo
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    #bucle que va actualizando fotograma por fotograma
    while running:
        ret, frame = cap.read()
        if not ret: break

        #voltear la imagen
        frame = cv2.flip(frame, 1)

        #configuración de la pantalla
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #devuelve los puntos de la mano
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                
                #dibujar las conexiones
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if label == "Left":
                    indice_arriba = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                    indice_abajo = hand_landmarks.landmark[12].y > hand_landmarks.landmark[6].y
                    medio_abajo = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
                    menique_abajo = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
                    anular_abajo = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y

                    if indice_arriba and medio_abajo and menique_abajo and anular_abajo:
                        if not pygame_ventana_abierta:
                            pg.init()
                            #screen = pg.display.set_mode((400, 300))
                            #pg.display.set_caption("Activado por Mano Izquierda")
                            #pygame_ventana_abierta = True
                            if bandera:
                                webbrowser.open(url)
                                bandera = False
                    elif indice_abajo and medio_abajo and menique_abajo and anular_abajo:
                        bandera = True

                
        if pygame_ventana_abierta:
            screen.fill((50, 150, 255)) 
            pg.display.update()
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    pygame_ventana_abierta = False

        frame = cv2.resize(frame, (WIDTH, HEIGTH))
        cv2.imshow('Detector de Gestos', frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            running = False

cap.release()
cv2.destroyAllWindows()
if pygame_ventana_abierta: pg.quit()
