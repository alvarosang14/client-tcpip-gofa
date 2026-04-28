#!/usr/bin/env python3

import socket

class GofaSocket:
    def __init__(self, host="192.168.125.1", port=1025):
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.connect((self.host, self.port))

        print("Recibido: ", self.receive_data().decode())

    def send_data(self, data: bytes):
        self.socket.sendall(data)

    def receive_data(self, buffer_size=1024):
        return self.socket.recv(buffer_size)
    
    def close(self):
        self.socket.close()