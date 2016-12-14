

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType, FloatType}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

val sqlContext = new SQLContext(sc)

val NUM_CLUSTERS = 200
val NUM_ITERATIONS = 5

val customSchema = StructType(Array(StructField("index", IntegerType, true),
  StructField("y", DoubleType, true), StructField("x", DoubleType, true)))

val df = {
  sqlContext.read.format("com.databricks.spark.csv").option("header","true").schema(customSchema).load("sample_coordinates.csv").limit(200000)
}

val parsed_df = {
  df.rdd.map(s => Vectors.dense(s.getDouble(1), s.getDouble(2))).cache()
}

val clusters = KMeans.train(parsed_df, NUM_CLUSTERS, NUM_ITERATIONS)

val WSSSE = clusters.computeCost(parsed_df)

clusters.save(sc, "target/org/apache/spark/coordinates/KMeansModel")

System.exit(0) 


