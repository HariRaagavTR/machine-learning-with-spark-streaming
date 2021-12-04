#!/usr/bin/env python3

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sparkContext = SparkContext.getOrCreate()
sparkStreamingContext = StreamingContext(sparkContext, 1)

# TCP_IP = sys.argv[1]
# TCP_PORT = int(sys.argv[2])

TCP_IP = 'localhost'
TCP_PORT = 6100

dataStream = sparkStreamingContext.socketTextStream(TCP_IP, TCP_PORT)
dataStream.pprint()

sparkStreamingContext.start()
sparkStreamingContext.awaitTermination()