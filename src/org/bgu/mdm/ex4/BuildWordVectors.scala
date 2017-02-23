package org.bgu.mdm.ex4

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.streaming._
import scala.collection._
import java.io._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.{ CoordinateMatrix, MatrixEntry }
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.mllib.optimization._

object BuildWordVectors extends App {

  val savedRDDfiles = "coocMatrix";
  val savedVectorFiles = "wordVectors";

  val logFile = "app2.log" // optional log file 
  val VECTOR_SIZE = 50 // גודל הווקטור לייצוג המילה
  val MAX_ITER = 15 // מספר איטרציות
  val X_MAX = 100 //  ערכו של X_MAX בנוסחה 9 במאמר
  val Alpha = 0.75 // ערכו של פרמטר Alpha בנוסחה 9 במאמר

  var startTime = System.nanoTime();

  // helper function to log operations to a file
  def log(str: String) = {
    if (!logFile.isEmpty()) {
      val out = new PrintWriter(new BufferedWriter(new FileWriter(logFile, true)));
      out.println(((System.nanoTime() - startTime) / 1000000000).toString + " : " + str);
      out.close();
    }
  }

  startTime = System.nanoTime();
  log("============ Start Run ========");

  // initialize spark and spark-sql
  val conf = new SparkConf().setAppName("CoClustering").setMaster("local")
  val sc = new SparkContext(conf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._

  // Read the cooccurrence matrix
  // -----------------------------

  log("Start word vectors, reading from " + savedRDDfiles);

  // restore saved RDD's
  val wordKeys = sc.objectFile[(String, Long)](savedRDDfiles + "Keys")
  val counts = sc.objectFile[((Long, Long), Int)](savedRDDfiles)
  // convert counts to matrix format
  val wordsMatrix = new CoordinateMatrix(counts.map(item => new MatrixEntry(item._1._1, item._1._2, item._2)));

  // Build the words vectors
  // -----------------------------
  val updater = new SimpleUpdater() // do we need L1 / L2 updater?
  val gradient = new LeastSquaresGradient() // do we need a custom gradient?
  GradientDescent.runMiniBatchSGD(data, gradient, updater, stepSize, MAX_ITER, regParam, miniBatchFraction, initialWeights)
  
  
  log("End word vectors, saving to " + savedVectorFiles);


  log("============ End Run ========");

}