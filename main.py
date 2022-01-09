#!/usr/bin/env python3
# Command: <path-to-spark-submit> main.py [-t] [-m <modelType>]

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
import numpy as np
import json
import argparse
from analysis import preprocess, train, test

parser = argparse.ArgumentParser(description = 'Receives data from a TCP socket and trains ML models.')

# Testing Argument. Default: False
parser.add_argument(
    '--test', '-t',
    help = 'Test Model?',
    required = False,
    type = bool,
    default = False
)

# Model Name Argument. Default: LRClassifier
parser.add_argument(
    '--model', '-m',
    help = 'Model Name',
    required = False,
    type = str,
    default = 'LRClassifier'
)

args = parser.parse_args()
testData = args.test
modelType = args.model

# Global variable of all unique classes.
allLabels = np.asarray([])

if modelType not in ['MNBClassifier', 'BNBClassifier', 'PAClassifier', 'SVMClassifier', 'PerceptronClassifier']:
    print('Critical Error: Invalid Model Name. Exiting Program.')
    exit()

def convertBatchToArray(batch):
    """
    Function that converts every batch into a numpy array which is used for further analysis.
    Args:
        batch (pyspark.rdd.RDD): A batch read from the spark stream.
    Returns:
        tuple(numpy.array, numpy.array): The batch data (X, Y).
    """
    try:
        data = json.loads(batch.collect()[0])
        images = []
        labels = []
        
        for index in data:
            record = data[index]
            # Preprocessing each image directly.
            preprocessedImage = preprocess(np.asarray(record["image"]))
            images.append(preprocessedImage)
            labels.append(record['label'])
        images = np.asarray(images)
        labels = np.asarray(labels)
        np.union1d(allLabels, np.unique(labels))
        
        return (images, labels)
    
    except:
        print('Error: Empty or Invalid Batch Received.')
        return (np.asarray([]), np.asarray([]))
    
def processBatch(batch):
    """
    Function that does the following:
    1. Convert received batch (training data) into a numpy array.
    2. Preprocess the numpy array.
    3. Incrementally trains 3 different models using the preprocessed data.

    Args:
        batch (pyspark.rdd.RDD): A batch read from the spark stream.
    Returns:
        N/A.
    """
    X, Y = convertBatchToArray(batch)
    if not testData:
        train(X, Y, modelType, allLabels)
    else:
        test(X, Y, modelType)

TCP_IP = 'localhost'
TCP_PORT = 6100

sparkContext = SparkContext("local[*]")
sparkStreamingContext = StreamingContext(sparkContext, 1)
sparkSQLContext = SQLContext(sparkContext)

dataStream = sparkStreamingContext \
    .socketTextStream(TCP_IP, TCP_PORT) \
    .foreachRDD(processBatch)

sparkStreamingContext.start()
sparkStreamingContext.awaitTermination(1000)
sparkStreamingContext.stop()