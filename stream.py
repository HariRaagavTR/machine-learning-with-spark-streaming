#!/usr/bin/python3
# Command: python3 stream.py [-b <batchSize>] [-e <True/False>]

import time
import json
import pickle
import socket
import argparse
from tqdm.auto import tqdm

TCP_IP = "localhost"
TCP_PORT = 6100

parser = argparse.ArgumentParser(description = 'Streams to Spark Streaming Context')

# Batch Size Argument. Default: 100
parser.add_argument(
    '--batch-size', '-b',
    help = 'Batch Size',
    required = False,
    type = int,
    default = 100
)

# Endless Stream Argument. Default: False
parser.add_argument(
    '--endless', '-e',
    help = 'Enable Endless Stream',
    required=False,
    type = bool,
    default = False
)


def sendPokemonBatchFileToSpark(tcpConnection, inputBatchFile):
    """
    Function that sends the dataset in batches to the spark client.

    Args:
        tcpConnection ([type]): Reference to TCP connection socket to the client.
        inputBatchFile (str): Batch file name.
    Returns:
        N/A.
    """
    # Loading entire dataset.
    with open(f'pokemon/{inputBatchFile}.pickle', 'rb') as batch_file:
        batchData = pickle.load(batch_file)
    
    data = batchData['img']
    labels = batchData['label']
    
    # Iterating over batches of size = batchSize.
    for imageIndex in tqdm(range(0, len(data) - batchSize + 2, batchSize)):
        imageDataBatch = data[imageIndex : imageIndex + batchSize]
        imageLabels = labels[imageIndex : imageIndex + batchSize] 
        
        payload = dict()
        for batchIndex in range(len(imageDataBatch)):   
            payload[batchIndex] = dict()  
            payload[batchIndex]['img'] = imageDataBatch[batchIndex]
            payload[batchIndex]['label'] = imageLabels[batchIndex]
        # print(payload)

        batchAsString = json.dumps(payload).encode()
        
        try:
            tcpConnection.send(batchAsString + '\n')
        except BrokenPipeError:
            print("Error: Either batch size is too big or the connection was closed.")
        except Exception as error_message:
            print(f"Error: {error_message}.")
            
        time.sleep(5)
            

def streamPokemonDataset(tcpConnection):
    """
    Function that iterates over all batch files and sends them to the client
    using the sendPokemonBatchFileToSpark() function.

    Args:
        tcpConnection (socket.socket): Reference to TCP connection socket to the client.
    Returns:
        N/A.
    """
    pokemonBatches = [
        'train_batch_1',
        # 'train_batch_2',
        # 'train_batch_3',
        # 'train_batch_4',
        # 'train_batch_5',
        # 'test_batch'
    ]
    print("Info: Streaming Pokemon Data.")
    for batch in pokemonBatches:
        # sendPokemonBatchFileToSpark(tcpConnection, batch)
        time.sleep(5)


def connectTCP():
    """
    Function that does the following:
    1. Create a TCP socket.
    2. Listen for a client.
    3. Connects the client to a new socket and returns the connection reference.
    
    Args:
        N/A.
    Returns:
        tuple(socket.socket, tuple(str, int)): Client connection socket reference and client address.
    """
    socketRef = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socketRef.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socketRef.bind((TCP_IP, TCP_PORT))
    socketRef.listen(1)
    
    print(f"Info: Waiting for TCP connection on port #{TCP_PORT}.")
    
    connection, address = socketRef.accept()
    print(f"Info: Connected to {address}.")

    return connection, address


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    batchSize = args.batch_size
    print(batchSize)
    endless = args.endless

    tcpConnection, _ = connectTCP()
    
    if endless:
        while True:
            streamPokemonDataset(tcpConnection)
    else:
        streamPokemonDataset(tcpConnection)

    tcpConnection.close()