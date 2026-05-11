#!/usr/bin/env python3

from camera.camera import Camera
from gofa_socket.gofa_socket import GofaSocket

import sys

class Engine:
    def __init__(self, mode):
        if mode:
            self.socket = GofaSocket()
            self.camera = Camera(self.socket, port_camera=2)
        else:
            self.camera = Camera(None, port_camera=2)

    def run(self):
        self.camera.run()

        self.camera.close()

        if self.camera.gofa_socket:
             self.camera.gofa_socket.close()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Uso: python main.py <mode>")
        sys.exit(1)
    elif len(sys.argv) > 1:
        print("Modo con socket")
        mode = True
    else:        
        print("Modo sin socket")
        mode = False

    engine = Engine(mode)
    engine.run()