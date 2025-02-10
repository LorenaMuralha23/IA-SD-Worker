import random
import socket
import struct
import json

from worker import receiveTask, createJson


MULTICAST_GROUP = '224.1.1.1'
PORT = 5007
MACHINE_ID = 0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                1)

sock.bind(('', PORT))

mreq = struct.pack("4s4s", socket.inet_aton(
    MULTICAST_GROUP), socket.inet_aton("0.0.0.0"))
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

print("Cliente Multicast aguardando mensagens...")


def receiveMessage():
    cont = 0
    while (True):
        data, addr = sock.recvfrom(4096)
        try:
            decoded_data = data.decode()
            receivedJson = json.loads(decoded_data.strip())

            if (type(receivedJson) is dict):
                if receivedJson.get("machine_id") != MACHINE_ID:  # <-
                    receiveTask(receivedJson)

            else:
                print("O JSON recebido não é um dicionário válido.")

            cont += 1
        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON de {addr}: {data.decode()}")


def sendToGroup(taskToSend):
    print("Json a ser enviado: " + json.dumps(taskToSend))
    sock.sendto(taskToSend.encode(), (MULTICAST_GROUP, PORT))
    receiveMessage()


def generate_unique_id():
    return random.randint(1, 10000)


if __name__ == "__main__":
    MACHINE_ID = generate_unique_id()
    print("ID GERADO: " , MACHINE_ID)
    createdJson = createJson("ONLINE", MACHINE_ID)
    sendToGroup(createdJson)
