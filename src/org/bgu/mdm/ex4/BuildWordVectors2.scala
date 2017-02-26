package org.bgu.mdm.ex4

import java.io.BufferedWriter
import java.io.FileWriter
import java.io.PrintWriter

import scala.util.Random

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.GloveGradient
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.GradientDescent
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.rdd.RDD

object BuildWordVectors2 extends App {

  val savedRDDfiles = "coocMatrix";
  val savedVectorFiles = "wordVectors";

  val logFile = "app4.log" // optional log file 
  val VECTOR_SIZE = 50 // גודל הווקטור לייצוג המילה
  val stepSize = 0.1
  val MAX_ITER = 15 // מספר איטרציות
  val miniBatchFraction = 0.4

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
  
  // Read the cooccurrence matrix
  // -----------------------------

  log("Start word vectors, reading from " + savedRDDfiles);

  // restore saved RDD's
  val wordKeys = sc.objectFile[(String, Long)](savedRDDfiles + "Keys")
  val counts = sc.objectFile[((Long, Long), Int)](savedRDDfiles)

  val vocabularySize = wordKeys.count().toInt;
  val maxWordId = wordKeys.map( _._2).reduce( (a,b) => ( if (a>b) a else b ) )
  log("vocabularySize is " + vocabularySize)
  log("maxWordId is " + maxWordId)

  // convert counts to matrix format from RDD of wordId,wordId,count
  // to RDD of count, vector of the two word id's
  val wordsMatrix: RDD[scala.Tuple2[Double, Vector]] = counts.map(item => (item._2, Vectors.dense(item._1._1, item._1._2)));

  // Build the words vectors
  // -----------------------------
  val updater = new SimpleUpdater()
  val gradient = new GloveGradient()
  val initialWeights: Vector = Vectors.dense(Array.fill((maxWordId.toInt+1) * (VECTOR_SIZE+1) )(Random.nextDouble))
  val result = GradientDescent.runMiniBatchSGD(wordsMatrix, gradient, updater, stepSize, MAX_ITER, 0, miniBatchFraction, initialWeights)

  log("End word vectors, saving to " + savedVectorFiles);

  sc.parallelize(result._1.toArray).saveAsObjectFile(savedVectorFiles)
  sc.parallelize(result._2).saveAsObjectFile(savedVectorFiles + "Loss")

  log("loss per iteration: " + result._2)

  log("============ End Run ========");

}