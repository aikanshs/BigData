import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object tweets {
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage: Tweets InputDir OutputDir")
    }

    val spark = SparkSession
      .builder
      .appName("Tweet Sentiment")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    val input=spark.read.option("header","true").option("inferSchema","true").csv(args(0))

    val filtered = input.filter(col("text") =!= "null")

    val Array(train, test) = filtered.randomSplit(Array(0.9, 0.1))

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("swr_text")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")
    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
      .setFeaturesCol(hashingTF.getOutputCol)
      .setLabelCol(indexer.getOutputCol)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, indexer, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.03, 0.001))
      .addGrid(lr.maxIter, Array(10, 20))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val cvModel = cv.fit(train)

    val prediction = cvModel.transform(test)

    val predictionAndLabels = prediction.select("label", "prediction").rdd.map(r => (r(0).asInstanceOf[Double], r(1).asInstanceOf[Double]))

    val metrics = new MulticlassMetrics(predictionAndLabels)

    var result="Accuracy: " + metrics.accuracy+"\n"

    result += "Confusion matrix:"+"\n"
    result += metrics.confusionMatrix.toString()+"\n"

    val labels = metrics.labels
    labels.foreach { l =>
      result += s"Precision($l) = " + metrics.precision(l)+"\n"
    }

    labels.foreach { l =>
      result += s"Recall($l) = " + metrics.recall(l)+"\n"
    }
    labels.foreach { l =>
      result += s"F1-Score($l) = " + metrics.fMeasure(l)+"\n"
    }

    labels.foreach { l =>
      result += s"Weighted F1-Score($l) = " + metrics.weightedFMeasure(l)+"\n"
    }

    val ans = spark.sparkContext.parallelize(List(result)).repartition(1)

    ans.saveAsTextFile(args(1))
  }
}
