#!/usr/bin/env python3
"""
Calibrador HSV con muestras positivas y negativas.

  r -> ROI del cubo que SI quieres detectar
  e -> ROI de lo que NO quieres detectar
Calcula los rangos H/S/V mas estrechos que cubren los positivos y
excluyen los negativos.

Teclas: r=pos | e=neg | z=undo | x=reset | p=print | +/-=area | s/ESC=salir
"""
import cv2
import numpy as np


class Calibrador:
    def __init__(self, port_camera=0, area_minima=1500,
                 margen=4, percentil=3, conservar_pos=0.95):
        self.cap = cv2.VideoCapture(port_camera)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara")
        try: self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception: pass

        self.area_minima, self.margen = area_minima, margen
        self.percentil, self.conservar_pos = percentil, conservar_pos
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.muestras_pos, self.muestras_neg, self.historial = [], [], []
        self.bajo = self.alto = None
        self.mouse = (0, 0)
        self.hsv_frame = self.frame_actual = None

    # ---------- Estado ----------
    def reset(self):
        self.muestras_pos.clear(); self.muestras_neg.clear(); self.historial.clear()
        self.bajo = self.alto = None
        print("\n>>> RESET <<<")

    def imprimir(self):
        if self.bajo is None:
            print("(no hay muestras positivas)"); return
        print(f"colorBajo = np.array([{self.bajo[0]}, {self.bajo[1]}, {self.bajo[2]}], np.uint8)")
        print(f"colorAlto = np.array([{self.alto[0]}, {self.alto[1]}, {self.alto[2]}], np.uint8)")

    def undo(self):
        if not self.historial:
            print("Nada que deshacer"); return
        tipo, _ = self.historial.pop()
        lista = self.muestras_pos if tipo == 'pos' else self.muestras_neg
        if lista: lista.pop()
        self.recalcular()
        print(f"Deshecha ultima muestra ({tipo})")

    # ---------- Muestreo ----------
    def _tomar_roi(self, titulo):
        if self.frame_actual is None: return None
        x, y, w, h = cv2.selectROI(titulo, self.frame_actual,
                                   showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(titulo)
        if w < 3 or h < 3: return None
        return self.hsv_frame[y:y+h, x:x+w].reshape(-1, 3).astype(np.int32)

    def muestrear(self, tipo):
        pos = (tipo == 'pos')
        lista = self.muestras_pos if pos else self.muestras_neg
        titulo = ("ROI POSITIVO: el cubo que SI quieres - ENTER" if pos
                  else "ROI A EXCLUIR: lo que NO quieres - ENTER")
        m = self._tomar_roi(titulo)
        if m is None:
            print("ROI cancelado o muy pequenyo"); return
        lista.append(m)
        self.historial.append((tipo, len(lista) - 1))
        print(f"\n[{'+' if pos else '-'} {tipo} #{len(lista)}]  ({len(m)} pixeles)")
        self.recalcular()

    # ---------- Calculo de bounds ----------
    def recalcular(self):
        if not self.muestras_pos:
            self.bajo = self.alto = None; return

        pos = np.vstack(self.muestras_pos)
        neg = (np.vstack(self.muestras_neg) if self.muestras_neg
               else np.empty((0, 3), np.int32))

        # 1) Bounds base: percentiles del positivo + margen
        pos_low  = np.percentile(pos, self.percentil,       axis=0)
        pos_high = np.percentile(pos, 100 - self.percentil, axis=0)
        pos_med  = np.median(pos, axis=0)
        bajo = (pos_low  - self.margen).astype(np.int32)
        alto = (pos_high + self.margen).astype(np.int32)

        # 2) Apretar bounds para excluir negativos sin perder >5% de positivos
        if len(neg):
            for c in range(3):
                n_c, p_c, med = neg[:, c], pos[:, c], pos_med[c]
                debajo = n_c[n_c < med]
                if len(debajo):
                    cand = int(debajo.max()) + 1
                    if cand <= alto[c] and (p_c >= cand).mean() >= self.conservar_pos:
                        bajo[c] = max(bajo[c], cand)
                encima = n_c[n_c > med]
                if len(encima):
                    cand = int(encima.min()) - 1
                    if cand >= bajo[c] and (p_c <= cand).mean() >= self.conservar_pos:
                        alto[c] = min(alto[c], cand)
                if bajo[c] > alto[c]:  # patologico: vuelvo al base
                    bajo[c] = int(pos_low[c])  - self.margen
                    alto[c] = int(pos_high[c]) + self.margen

        self.bajo = np.clip(bajo, 0, [179, 255, 255]).astype(np.uint8)
        self.alto = np.clip(alto, 0, [179, 255, 255]).astype(np.uint8)
        self.imprimir()

        if len(neg):
            pd = ((pos >= self.bajo) & (pos <= self.alto)).all(axis=1).mean()
            nd = ((neg >= self.bajo) & (neg <= self.alto)).all(axis=1).mean()
            print(f"  Positivos cubiertos: {pd*100:.1f}%   "
                  f"Negativos cubiertos: {nd*100:.1f}%")
            if nd > 0.01:
                print("  AVISO: no se han excluido todos los negativos.")

    # ---------- Filtrado ----------
    def filtrar_area(self, mask):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        validos = [c for c in contornos if cv2.contourArea(c) > self.area_minima]
        mask_lim = np.zeros_like(mask)
        cv2.drawContours(mask_lim, validos, -1, 255, -1)
        return mask_lim, validos

    # ---------- Bucle principal ----------
    def run(self):
        cv2.namedWindow('Calibrador', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Calibrador',
                             lambda ev, x, y, *_: setattr(self, 'mouse', (x, y)))

        acciones = {
            ord('r'): lambda: self.muestrear('pos'),
            ord('e'): lambda: self.muestrear('neg'),
            ord('z'): self.undo,
            ord('x'): self.reset,
            ord('p'): self.imprimir,
        }

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            self.frame_actual = frame
            self.hsv_frame = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0),
                                          cv2.COLOR_BGR2HSV)
            vista = frame.copy()

            mx, my = self.mouse
            if 0 <= my < self.hsv_frame.shape[0] and 0 <= mx < self.hsv_frame.shape[1]:
                h, s, v = self.hsv_frame[my, mx]
                cv2.circle(vista, (mx, my), 10, (0, 255, 255), 1)
                cv2.putText(vista, f"HSV cursor: ({h}, {s}, {v})", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if self.bajo is not None:
                mask = cv2.inRange(self.hsv_frame, self.bajo, self.alto)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
                mask, validos = self.filtrar_area(mask)
                vista[mask == 0] = (vista[mask == 0] * 0.2).astype(np.uint8)
                cv2.drawContours(vista, validos, -1, (0, 255, 0), 2)
                cv2.putText(vista,
                    f"Pos:{len(self.muestras_pos)} Neg:{len(self.muestras_neg)} "
                    f"Blobs:{len(validos)} "
                    f"H[{self.bajo[0]}-{self.alto[0]}] "
                    f"S[{self.bajo[1]}-{self.alto[1]}] "
                    f"V[{self.bajo[2]}-{self.alto[2]}]  area>{self.area_minima}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.putText(vista,
                "r=pos | e=neg | z=undo | x=reset | p=print | +/-=area | s/ESC=salir",
                (10, vista.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

            cv2.imshow('Calibrador', vista)
            key = cv2.waitKey(30) & 0xFF
            if key in acciones:
                acciones[key]()
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
    Calibrador(port_camera=0).run()