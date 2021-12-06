#!/usr/bin/env python3
from pyspark.sql import SparkSession

TCP_IP = 'localhost'
TCP_PORT = 6100
BATCH_SIZE = 10

spark = SparkSession \
    .builder \
    .appName("Image Classifier") \
    .getOrCreate()

dfFromStream = spark.readStream \
      .format("socket") \
      .option("host", TCP_IP) \
      .option("port", TCP_PORT) \
      .load()

# schema =  StructType([
#     StructField("img", StringType(), True),
#     StructField("label", StringType(), True)
# ])

# convertedDataFrame = dfFromStream \
#     .withColumn("jsonData",from_json(col("value"), schema)) \
#     .select("jsonData.*")
    
query = dfFromStream.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()
    
query.awaitTermination()

"""
"0": {
    "img": [[[]]]
    "label": "className"
},
"1": {
    "img": [[[]]]
    "label": "className"
},
"2": {
    "img": [[[]]]
    "label": "className"
},
"""