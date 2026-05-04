#!/usr/bin/env python3

import cv2
import numpy as np
import time

class Camera:
    def __init__(self, gofa_socket=None, port_camera=0):
        self.gofa_socket = gofa_socket

        self.PIXELS_POR_MM = 16.43
        self.port_camera = port_camera

        self.cap = cv2.VideoCapture(self.port_camera)

        # Rojo bajo (tonos cerca de H=0)
        self.redBajo1 = np.array([  0, 101, 153], np.uint8)
        self.redAlto1 = np.array([ 10, 197, 235], np.uint8)
        # Rojo alto (tonos cerca de H=179)
        self.redBajo2 = np.array([170, 101, 153], np.uint8)
        self.redAlto2 = np.array([179, 197, 235], np.uint8)

        self.blancoBajo = np.array([103, 38, 195], np.uint8)
        self.blancoAlto = np.array([118, 67, 255], np.uint8)

        self.celesteBajo = np.array([95, 183, 180], np.uint8)
        self.celesteAlto = np.array([105, 236, 253], np.uint8)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def dibujar(self, mask, color, label):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 3000:
                M = cv2.moments(c)
                if M["m00"] == 0: M["m00"] = 1
                x = int(M["m10"] / M["m00"])
                y = int(M['m01'] / M["m00"])
                nuevoContorno = cv2.convexHull(c)
                cv2.circle(self.frame, (x, y), 7, (0, 255, 0), -1)

                x_mm = round(x / self.PIXELS_POR_MM, 1)
                y_mm = round(y / self.PIXELS_POR_MM, 1)

                cv2.putText(self.frame, '{:.1f}mm,{:.1f}mm'.format(x_mm, y_mm), 
                    (x + 10, y), self.font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.drawContours(self.frame, [nuevoContorno], 0, color, 3)

                # No va a ver mas de tres digitos sino seria un metro
                coorX = '{:04.1f}'.format(x_mm)
                coorY = '{:04.1f}'.format(y_mm)
                coordXY = label + coorX + coorY

                if self.gofa_socket:
                    self.gofa_socket.send_data(coordXY.encode())

                time.sleep(1)
                print("Coordenadas: " + coordXY + " -> {:.1f}mm, {:.1f}mm, Color: {}".format(x_mm, y_mm, label))

    def run(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if ret == True:
                self.frame = frame
                frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                maskBlanco = cv2.inRange(frameHSV, self.blancoBajo, self.blancoAlto)
                maskCeleste = cv2.inRange(frameHSV, self.celesteBajo, self.celesteAlto)
                maskRed1   = cv2.inRange(frameHSV, self.redBajo1, self.redAlto1)
                maskRed2   = cv2.inRange(frameHSV, self.redBajo2, self.redAlto2)
                maskRed    = cv2.add(maskRed1, maskRed2)
                self.dibujar(maskBlanco,  (255, 255, 255), 'W')   # Blanco
                self.dibujar(maskCeleste, (255, 255,   0), 'B')   # Celeste = azul
                self.dibujar(maskRed,     (  0,   0, 255), 'R')   # Rojo
                cv2.imshow('frame', self.frame)

            key = cv2.waitKey(30) & 0xFF
            window_closed = cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0
            if key == ord('s') or key == 27 or window_closed:
                self.close()
                if self.gofa_socket:
                    self.gofa_socket.close()
                print("Saliendo...")
                break

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()