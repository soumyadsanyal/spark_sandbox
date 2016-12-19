import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType, FloatType}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import java.util.Calendar

val sqlContext = new SQLContext(sc)

val NUM_CLUSTERS = 2
val NUM_ITERATIONS = 5

val customSchema = StructType(Array(StructField("index", IntegerType, true),
  StructField("y", DoubleType, true), StructField("x", DoubleType, true)))

val df = {
  sqlContext.read.format("com.databricks.spark.csv").option("header","true").schema(customSchema).load("sample_coordinates.csv").limit(200)
}

val parsed_df = {
  df.rdd.map(s => Vectors.dense(s.getDouble(1), s.getDouble(2))).cache()
}

val clusters = KMeans.train(parsed_df, NUM_CLUSTERS, NUM_ITERATIONS)

val WSSSE = clusters.computeCost(parsed_df)

val time_index = Calendar.getInstance().getTime.toString.map(c => if (c==' ') '_' else c)

clusters.save(sc, s"target/org/apache/spark/coordinates_all_data/KMeansModel_$time_index")

val res = clusters.clusterCenters

val lats = res.map(v => v(0))
val lons = res.map(v => v(1))
val idx = 1 to lats.length

case class RecordClass(idx: Int, lat: Double, lon: Double)

val l = List[Seq[Any]](idx, lats, lons).map(x => x.length).min

val res_coords = (0 until l) map {
  i => RecordClass(idx(i), lats(i), lons(i))
}

val centers = sc.parallelize(res_coords).toDF

centers.write.format("com.databricks.spark.csv").save(s"./thecenters_$time_index.csv")

/* System.exit(0) */


