import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object PageRank {
  def main(args: Array[String]): Unit = {

    if (args.length != 3) {
      println("Usage: InputDir Iterations OutputDir")
    }
    val spark = SparkSession
      .builder
      .appName("Page Rank")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    var airports = spark.read.option("header","true").option("inferSchema","true").csv(args(0))
    var pagerank = spark.read.option("header","true").option("inferSchema","true").csv(args(0))

    val alpha = 0.15
    val df_count = alpha / airports.count()

    val iterations = args(1).toInt
    airports = airports.withColumn("PAGE_RANK",lit(10.0))

    for(_ <- 1 to iterations){
      val outDegree = airports.groupBy("ORIGIN_AIRPORT_ID").agg(count("ORIGIN_AIRPORT_ID") as "outlinks")
      airports = airports.join(outDegree,Seq("ORIGIN_AIRPORT_ID"))
      airports = airports.withColumn("linkedAirports", col("PAGE_RANK") / col("outlinks"))
      pagerank = airports.groupBy("DEST_AIRPORT_ID").agg((lit(df_count)+(lit(1-alpha)*sum("linkedAirports"))) as "PAGE_RANK")
      airports = airports.select("ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID")
      airports = airports.join(pagerank, Seq("DEST_AIRPORT_ID"))
    }
    pagerank = pagerank.sort(desc("PAGE_RANK"))
    val result = pagerank.repartition(1)
    result.write.format("csv").save(args(2))
  }
}
