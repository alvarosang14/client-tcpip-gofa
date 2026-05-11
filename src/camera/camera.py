#!/usr/bin/env python3

#     SocketSend clientSocket \Str:="GET "+color_a_buscar+"\0A";
#!/usr/bin/env python3
import cv2
import numpy as np

class Camera:
    def __init__(self, gofa_socket=None, port_camera=0):
        self.gofa_socket = gofa_socket
        self.PIXELS_POR_CM = 16.8
        self.port_camera = port_camera
        self.cap = cv2.VideoCapture(self.port_camera)
        # Rojo bajo (tonos cerca de H=0)
        self.redBajo1 = np.array([  0, 101, 153], np.uint8)
        self.redAlto1 = np.array([ 10, 197, 235], np.uint8)
        # Rojo alto (tonos cerca de H=179)
        self.redBajo2 = np.array([170, 101, 153], np.uint8)
        self.redAlto2 = np.array([179, 197, 235], np.uint8)
        # Blanco
        self.blancoBajo = np.array([103, 38, 195], np.uint8)
        self.blancoAlto = np.array([118, 67, 255], np.uint8)
        # Celeste
        self.celesteBajo = np.array([95, 183, 180], np.uint8)
        self.celesteAlto = np.array([105, 236, 253], np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Diccionario de candidatos
        self.candidatos = {}
        self.buf_rx = b""
        if self.gofa_socket:
            self.gofa_socket.socket.setblocking(False)

    def check_messages(self):
        """
        Lee mensajes del socket y responde con coordenadas de los candidatos
        """
        if not self.gofa_socket:
            return
        try:
            while True:
                data = self.gofa_socket.socket.recv(64)
                if not data:
                    break
                self.buf_rx += data
        except BlockingIOError:
            pass
        except OSError:
            pass

        while b"\n" in self.buf_rx:
            print("Received raw data:", self.buf_rx)
            linea, self.buf_rx = self.buf_rx.split(b"\n", 1)
            msg = linea.strip().decode(errors="ignore")
            if msg.startswith("Get"):
                print("Received message:", msg)
                partes = msg.split()
                color = partes[1] if len(partes) > 1 else None
                if color and color in self.candidatos:
                    x_mm, y_mm = self.candidatos[color]
                    coordXY = "{}{:04.0f}{:04.0f}\n".format(color, x_mm, y_mm)
                    self.gofa_socket.send_data(coordXY.encode())
                    print("Petición {} -> envío {}".format(color, coordXY.strip()))
                else:
                    self.gofa_socket.send_data(b"NONE\n")
                    print("Petición {} -> no hay candidato".format(color))

    def dibujar(self, mask, color, label):
        """
        Dibuja contornos y calcula coordenadas de candidatos
        """
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
                x_mm = round(x / self.PIXELS_POR_CM, 1) * 10
                y_mm = round(y / self.PIXELS_POR_CM, 1) * 10
                cv2.putText(self.frame, '{:.1f}mm,{:.1f}mm'.format(x_mm, y_mm),
                    (x + 10, y), self.font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.drawContours(self.frame, [nuevoContorno], 0, color, 3)

                self.candidatos[label] = (x_mm, y_mm)

    def run(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if ret == True:
                self.frame = frame
                # Reset en cada frame
                self.candidatos = {}
                frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                maskBlanco = cv2.inRange(frameHSV, self.blancoBajo, self.blancoAlto)
                maskCeleste = cv2.inRange(frameHSV, self.celesteBajo, self.celesteAlto)
                maskRed1   = cv2.inRange(frameHSV, self.redBajo1, self.redAlto1)
                maskRed2   = cv2.inRange(frameHSV, self.redBajo2, self.redAlto2)
                maskRed    = cv2.add(maskRed1, maskRed2)
                self.dibujar(maskBlanco,  (255, 255, 255), 'B')
                self.dibujar(maskCeleste, (255, 255,   0), 'C')
                self.dibujar(maskRed,     (  0,   0, 255), 'R')
                cv2.imshow('frame', self.frame)

            # Revisar mensajes del socket
            self.check_messages()

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
