import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.graphframes.GraphFrame

object SocialNetwork {
    def main(args: Array[String]): Unit = {

        if (args.length != 2) {
            println("Usage: InputDir OutputDir")
        }

        val spark = SparkSession
          .builder
          .appName("Twitter Graph")
          .config("spark.some.config.option", "some-value")
          .getOrCreate()

        val input = spark.read.text(args(0))

        val data_temp = input.toDF
        val edges = data_temp.select(split(col("value"), " ").getItem(0).as("src"), split(col("value"), " ").getItem(1).as("dst"))

        val srcNodes = edges.select("src").distinct
        val dstNodes = edges.select("dst").distinct

        val nodes = srcNodes.union(dstNodes).withColumnRenamed("src", "id").distinct

        val twitterGraph = GraphFrame(nodes, edges)
        twitterGraph.cache()
        val d1=Row("Query 1")
        val qu1=spark.sparkContext.parallelize(Seq(d1))
        val q1 = twitterGraph.outDegrees.sort(desc("outDegree")).limit(5)
        val query1:RDD[Row] = q1.rdd

        val q2 = twitterGraph.inDegrees.sort(desc("inDegree")).limit(5)
        val query2:RDD[Row] = q2.rdd
        val d2=Row("Query 2")
        val qu2=spark.sparkContext.parallelize(Seq(d2))

        val ranks = twitterGraph.pageRank.resetProbability(0.15).maxIter(2).run()
        val q3 = ranks.vertices.orderBy(desc("pagerank")).select("id", "pagerank").limit(5)
        val query3:RDD[Row] = q3.rdd
        val d3= Row("Query 3")
        val qu3=spark.sparkContext.parallelize(Seq(d3))
        spark.sparkContext.setCheckpointDir("/tmp/checkpoints")

        val minGraph = GraphFrame(nodes, edges.sample(withReplacement = false, 0.1))
        val connComp = minGraph.connectedComponents.run()
        val q4 = connComp.where("component != 0").groupBy("component").count().sort(desc("count")).limit(5)
        val query4:RDD[Row] = q4.rdd
        val d4=Row("Query 4")
        val qu4=spark.sparkContext.parallelize(Seq(d4))

        val motifs = twitterGraph.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[ca]->(a)")
        val q5 = motifs.groupBy("a").count().withColumnRenamed("a", "id").sort(desc("count")).limit(5)
        val query5:RDD[Row] = q5.rdd
        val d5=Row("Query 5")
        val qu5=spark.sparkContext.parallelize(Seq(d5))

        val result=spark.sparkContext.union(qu1,query1,qu2,query2,qu3,query3,qu4,query4,qu5,query5)
        result.coalesce(1).saveAsTextFile(args(1))
    }
}
