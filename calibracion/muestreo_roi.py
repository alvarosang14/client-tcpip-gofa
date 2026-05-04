#!/usr/bin/env python3
"""
Calibrador HSV con muestras POSITIVAS y NEGATIVAS.

Idea clave: tu le dices al algoritmo
  - 'r' (ROI positivo): "esto SI lo quiero detectar"  (el cubo objetivo)
  - 'e' (ROI a excluir): "esto NO lo quiero detectar"  (los otros cubos)

El algoritmo calcula los rangos H/S/V mas estrechos posibles que:
  - Cubran el 95-100% de los pixeles positivos
  - EXCLUYAN todos los pixeles negativos

Asi, si solo seleccionas el cubo azul como positivo y marcas verde y rojo
como negativos, te garantiza que la mascara solo se enciende con el azul.

Flujo:
  1. Pasa el raton sobre los cubos sin clicar para diagnosticar HSV.
  2. 'r' -> ROI sobre el cubo que quieres detectar (puedes hacerlo varias
     veces en distintas posiciones/iluminaciones para acumular).
  3. 'e' -> ROI sobre cada cubo/objeto que NO quieres detectar.
  4. Cada vez que anyades algo, los rangos se recalculan automaticamente.
  5. 'p' para imprimir, 's'/ESC para salir.

Si los rangos no convergen (el algoritmo no encuentra forma de separar),
suele ser porque los colores son fisicamente parecidos en HSV o porque
metiste fondo/sombra en el ROI positivo.

Teclas:
  r  ROI positivo (cubo que quieres)
  e  ROI a excluir (cubo que NO quieres)
  z  deshacer ultima accion (positiva o negativa)
  x  reset total
  p  imprimir colorBajo/colorAlto
  +/-  area minima del filtro de blobs
  s/ESC  salir
"""
import cv2
import numpy as np


class Calibrador:
    def __init__(self, port_camera=0, area_minima=1500,
                 margen=4, percentil=3, conservar_pos=0.95):
        self.cap = cv2.VideoCapture(port_camera)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara")
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass

        self.area_minima   = area_minima
        self.margen        = margen
        self.percentil     = percentil
        self.conservar_pos = conservar_pos  # fraccion minima de positivos a mantener
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.muestras_pos = []   # lista de arrays HSV (cada uno de un ROI)
        self.muestras_neg = []
        self.historial    = []   # ('pos', idx) o ('neg', idx) para deshacer

        self.bajo = None
        self.alto = None
        self.mouse = (0, 0)
        self.hsv_frame = None
        self.frame_actual = None

    # ---------- Estado ----------
    def reset(self):
        self.muestras_pos.clear()
        self.muestras_neg.clear()
        self.historial.clear()
        self.bajo = None
        self.alto = None
        print("\n>>> RESET <<<")

    def imprimir(self):
        if self.bajo is None:
            print("(no hay muestras positivas)")
            return
        print(f"colorBajo = np.array([{self.bajo[0]}, {self.bajo[1]}, {self.bajo[2]}], np.uint8)")
        print(f"colorAlto = np.array([{self.alto[0]}, {self.alto[1]}, {self.alto[2]}], np.uint8)")

    def undo(self):
        if not self.historial:
            print("Nada que deshacer")
            return
        tipo, _ = self.historial.pop()
        if tipo == 'pos' and self.muestras_pos:
            self.muestras_pos.pop()
        elif tipo == 'neg' and self.muestras_neg:
            self.muestras_neg.pop()
        self.recalcular()
        print(f"Deshecha ultima muestra ({tipo})")

    # ---------- Mouse ----------
    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)

    # ---------- Muestreo ----------
    def _tomar_roi(self, titulo):
        if self.frame_actual is None:
            return None
        roi = cv2.selectROI(titulo, self.frame_actual,
                             showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(titulo)
        x, y, w, h = roi
        if w < 3 or h < 3:
            return None
        return self.hsv_frame[y:y+h, x:x+w].reshape(-1, 3).astype(np.int32)

    def muestrear_positivo(self):
        m = self._tomar_roi("ROI POSITIVO: el cubo que SI quieres - ENTER")
        if m is None:
            print("ROI cancelado o muy pequenyo")
            return
        self.muestras_pos.append(m)
        self.historial.append(('pos', len(self.muestras_pos) - 1))
        print(f"\n[+ Positivo #{len(self.muestras_pos)}]  ({len(m)} pixeles)")
        self.recalcular()

    def muestrear_negativo(self):
        m = self._tomar_roi("ROI A EXCLUIR: lo que NO quieres - ENTER")
        if m is None:
            print("ROI cancelado o muy pequenyo")
            return
        self.muestras_neg.append(m)
        self.historial.append(('neg', len(self.muestras_neg) - 1))
        print(f"\n[- Negativo #{len(self.muestras_neg)}]  ({len(m)} pixeles)")
        self.recalcular()

    # ---------- Calculo de bounds ----------
    def recalcular(self):
        if not self.muestras_pos:
            self.bajo = self.alto = None
            return

        pos = np.vstack(self.muestras_pos)
        neg = (np.vstack(self.muestras_neg)
               if self.muestras_neg else np.empty((0, 3), np.int32))

        # 1) Bounds base: percentiles del positivo
        pos_low  = np.percentile(pos, self.percentil,        axis=0)
        pos_high = np.percentile(pos, 100 - self.percentil,  axis=0)
        pos_med  = np.median(pos, axis=0)

        bajo = (pos_low  - self.margen).astype(np.int32)
        alto = (pos_high + self.margen).astype(np.int32)

        # 2) Si hay negativos, apretar los bounds para excluirlos
        if len(neg) > 0:
            for c in range(3):
                n_c = neg[:, c]

                # Negativos POR DEBAJO de la mediana del positivo:
                # subir bajo[c] hasta justo por encima del max de esos negativos
                debajo = n_c[n_c < pos_med[c]]
                if len(debajo) > 0:
                    candidato = int(debajo.max()) + 1
                    if candidato <= alto[c]:
                        # comprobar que conservamos suficientes positivos
                        if (pos[:, c] >= candidato).mean() >= self.conservar_pos:
                            bajo[c] = max(bajo[c], candidato)

                # Negativos POR ENCIMA de la mediana:
                # bajar alto[c] hasta justo por debajo del min
                encima = n_c[n_c > pos_med[c]]
                if len(encima) > 0:
                    candidato = int(encima.min()) - 1
                    if candidato >= bajo[c]:
                        if (pos[:, c] <= candidato).mean() >= self.conservar_pos:
                            alto[c] = min(alto[c], candidato)

        # Salvaguarda contra inversiones bajo > alto
        for c in range(3):
            if bajo[c] > alto[c]:
                # algun caso patologico: no puedo separar en este canal,
                # vuelvo al rango base
                bajo[c] = int(pos_low[c]) - self.margen
                alto[c] = int(pos_high[c]) + self.margen

        self.bajo = np.clip(bajo, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        self.alto = np.clip(alto, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        self.imprimir()

        # Reportar cuantos positivos y negativos quedan dentro
        if len(neg) > 0:
            pos_dentro = ((pos >= self.bajo) & (pos <= self.alto)).all(axis=1).mean()
            neg_dentro = ((neg >= self.bajo) & (neg <= self.alto)).all(axis=1).mean()
            print(f"  Positivos cubiertos: {pos_dentro*100:.1f}%   "
                  f"Negativos cubiertos: {neg_dentro*100:.1f}%  (objetivo: ~0%)")
            if neg_dentro > 0.01:
                print("  AVISO: no se ha podido excluir todos los negativos. "
                      "Anyade mas muestras o revisa los ROIs.")

    # ---------- Filtrado ----------
    def filtrar_area(self, mask):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        mask_lim = np.zeros_like(mask)
        validos = []
        for c in contornos:
            if cv2.contourArea(c) > self.area_minima:
                cv2.drawContours(mask_lim, [c], -1, 255, -1)
                validos.append(c)
        return mask_lim, validos

    # ---------- Bucle principal ----------
    def run(self):
        cv2.namedWindow('Calibrador', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibrador', self.on_mouse)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_actual = frame
            self.hsv_frame = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0),
                                           cv2.COLOR_BGR2HSV)
            vista = frame.copy()

            # HSV bajo cursor
            mx, my = self.mouse
            if (0 <= my < self.hsv_frame.shape[0]
                    and 0 <= mx < self.hsv_frame.shape[1]):
                h, s, v = self.hsv_frame[my, mx]
                cv2.circle(vista, (mx, my), 10, (0, 255, 255), 1)
                cv2.putText(vista, f"HSV cursor: ({h}, {s}, {v})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            # Mascara
            if self.bajo is not None:
                mask = cv2.inRange(self.hsv_frame, self.bajo, self.alto)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
                mask, validos = self.filtrar_area(mask)
                no = mask == 0
                vista[no] = (vista[no] * 0.2).astype(np.uint8)
                cv2.drawContours(vista, validos, -1, (0, 255, 0), 2)

                txt = (f"Pos: {len(self.muestras_pos)}  "
                       f"Neg: {len(self.muestras_neg)}  "
                       f"Blobs: {len(validos)}  "
                       f"H[{self.bajo[0]}-{self.alto[0]}] "
                       f"S[{self.bajo[1]}-{self.alto[1]}] "
                       f"V[{self.bajo[2]}-{self.alto[2]}]")
                cv2.putText(vista, txt, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(vista, f"Area minima: {self.area_minima} px",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)

            cv2.putText(vista,
                        "r=positivo | e=excluir | z=deshacer | x=reset | "
                        "p=imprimir | +/-=area | s/ESC=salir",
                        (10, vista.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

            cv2.imshow('Calibrador', vista)
            key = cv2.waitKey(30) & 0xFF
            if   key == ord('r'): self.muestrear_positivo()
            elif key == ord('e'): self.muestrear_negativo()
            elif key == ord('z'): self.undo()
            elif key == ord('x'): self.reset()
            elif key == ord('p'): self.imprimir()
            elif key in (ord('+'), ord('=')):
                self.area_minima += 250
                print(f"Area minima = {self.area_minima}")
            elif key == ord('-'):
                self.area_minima = max(100, self.area_minima - 250)
                print(f"Area minima = {self.area_minima}")
            elif key == ord('s') or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Calibrador(port_camera=2).run()