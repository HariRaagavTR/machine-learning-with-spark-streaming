#!/usr/bin/env python3
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
import numpy as np
import json
from analysis import preprocess

def convertBatchToArray(batch):
    """
    Function that converts every batch into a numpy array which is used for further analysis.
    Args:
        batch (pyspark.rdd.RDD): A batch read from the spark stream.
    Returns:
        numpy.array: The batch data represented as a numpy array.
    """
    try:
        data = json.loads(batch.collect()[0])
        values = []
        
        for index in data:
            record = data[index]
            # Preprocessing each image directly.
            image = preprocess(np.asarray(record['image']))
            label = record['label']
            values.append(np.asarray([image, label]))
        
        return np.asarray(values)
    
    except:
        print('Error: Empty or Invalid Batch Received.')
        return np.asarray([])
    
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
    values = convertBatchToArray(batch)
    print(values)

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