#!/usr/bin/env python3

from camera.camera import Camera
from gofa_socket.gofa_socket import GofaSocket


class Engine:
    def __init__(self):
        #self.socket = GofaSocket()
        #self.camera = Camera(self.socket)

        self.camera = Camera(None)

    def run(self):
        self.camera.run()

        self.camera.close()

        if self.camera.gofa_socket:
             self.camera.gofa_socket.close()

if __name__ == "__main__":
    engine = Engine()
    engine.run()