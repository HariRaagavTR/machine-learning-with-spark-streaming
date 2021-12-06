from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder \
    .appName('SpamDetection') \
    .getOrCreate()

def transformDataFrame(df):
    """
    Function that pre-processes the given dataframe.
    Args:
        df: DataFrame.
    Returns:
        The transformed dataframe.
    """
    # Splits the words of each tweet.
    tokenizer = Tokenizer (
        inputCol = 'Tweet',
        outputCol = 'Tweet-Words'
    )
    dfSplit = tokenizer.transform(df).drop('Tweet')

    # dfSplit.show(30)
    
    # Removes unimportant words.
    stopWordsRemover = StopWordsRemover (
        inputCol = 'Tweet-Words',
        outputCol = 'Important-Words'
    )
    dfImportant = stopWordsRemover.transform(dfSplit).drop('Tweet-Words')

    # dfImportant.show(30)
    
    # Converts important words into numerical data.
    hashTF = HashingTF (
        inputCol = 'Important-Words',
        outputCol = 'Converted-AAM3'
    )
    dfAAM3 = hashTF.transform(dfImportant).drop('Important-Words')

    # dfAAM3.show(30)
    
    return dfAAM3

def trainLogisticRegressionModel(trainingData):
    """
    Function that trains the model using Logistic Regression using training data.
    Args:
        trainingData: DataFrame.
    Returns:
        The model itself.
    """
    lr = LogisticRegression (
        labelCol = 'label',
        featuresCol = 'features',
        maxIter = 10,
        regParam = 0.01
    )
    model = lr.fit(trainingData)

    return model


# ------------------------------ TRAIN.CSV ------------------------------ #

trainingData = spark.read \
    .option('header', True) \
    .options(delimiter = ',') \
    .csv('./sentiment/train.csv')

# trainingData.show(30)
# print('Training Data Rows:', trainingData.count())

transformedTrainingData = transformDataFrame(trainingData)

trainingData = transformedTrainingData \
    .withColumnRenamed('Sentiment', 'label') \
    .withColumnRenamed('Converted-AAM3', 'features') \
    .select('features', col('label').cast('Int'))
trainingData.show(30)

# ------------------------------ TRAINING MODEL ------------------------------ #

model = trainLogisticRegressionModel(trainingData)

# ------------------------------ TEST.CSV ------------------------------ #

testingData = spark.read \
    .option('header', True) \
    .options(delimiter = ',') \
    .csv('./sentiment/test.csv')

# testingData.show(30, truncate=False)
# print('Testing Data Rows:', testingData.count())

transformedTestingData = transformDataFrame(testingData)
testingData = transformedTestingData \
    .withColumnRenamed('Sentiment', 'label') \
    .withColumnRenamed('Converted-AAM3', 'features') \
    .select('features', col('label').cast('Int'))
# testingData.show(20)

# ------------------------------ TESTING MODEL ------------------------------ #

prediction = model.transform(testingData).select('prediction', 'label')

# prediction.show(30)

# ------------------------------ ACCURACY ------------------------------ #

totalCorrect = prediction.filter(prediction['prediction'] == prediction['label']).count()
total = prediction.count()
print('Accuracy:', totalCorrect / total)