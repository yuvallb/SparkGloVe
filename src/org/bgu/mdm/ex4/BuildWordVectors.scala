package org.bgu.mdm.ex4

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql._
import org.apache.spark.streaming._
import scala.collection._
import java.io._
import scala.util.Random
import org.apache.spark.rdd.RDD
//import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.distributed.{ CoordinateMatrix, MatrixEntry }
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.mllib.optimization._
import breeze.linalg._

object BuildWordVectors extends App {

  val savedRDDfiles = "coocMatrixSmall";
  val savedVectorFiles = "wordVectorsSmall";
  //val savedRDDfiles = "coocMatrix";
  //val savedVectorFiles = "wordVectors";

  val logFile = "app3.log" // optional log file 
  val VECTOR_SIZE = 50 // גודל הווקטור לייצוג המילה
  val MAX_ITER = 15 // מספר איטרציות
  val X_MAX = 100 //  ערכו של X_MAX בנוסחה 9 במאמר
  val ALPHA = 0.75 // ערכו של פרמטר Alpha בנוסחה 9 במאמר
  val learningRate = 0.1

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

  val vocabularySize = wordKeys.count();

  // build w - RDD with key = word key, value = word vector  
  var w = wordKeys.map(word => (word._2, new DenseVector(Array.fill(VECTOR_SIZE)(Random.nextDouble - 0.5))))
  var W = sc.broadcast(w.collectAsMap())

  // build b - RDD with key = word key, value = word bias
  var b = wordKeys.map(word => (word._2, Random.nextDouble - 0.5))
  var B = sc.broadcast(b.collectAsMap())

  // weighting function - equation 9
  def weighting(x: Int): Double = {
    return if (x < X_MAX)
      Math.pow(x / X_MAX, ALPHA)
    else
      1
  }

  // calculate cost
  val cost0 = counts.map(pair => (
    weighting(pair._2) *
    Math.pow(W.value(pair._1._1).dot(W.value(pair._1._2)) +
      B.value(pair._1._1) + B.value(pair._1._1)
      - Math.log(pair._2), 2)))
    .reduce(_ + _)
  log("Initial cost: " + cost0)
  log("W size: " + W.value.size)
  log("B size: " + B.value.size)
  log("W 1 2 3: " + W.value(1) + " ; " + W.value(2) + " ; " + W.value(3) )
  log("B 1 2 3: " + B.value(1) + " ; " + B.value(2) + " ; " + B.value(3) )

  // optimization iterations
  for (iter <- 1 to MAX_ITER) {

    /*
      val gradients = counts.takeSample(false, 1).map( pair => {
       val inner_weight = W.value(pair._1._1).dot(W.value(pair._1._2))  +
        B.value(pair._1._1) + B.value(pair._1._1) +
          -Math.log(pair._2)  * weighting(pair._2)
        
        ( (pair._1._1 , pair._1._2) ,               // i , j
          W.value(pair._1._2) * inner_weight    // gradient i
        , W.value(pair._1._1) * inner_weight    // gradient j
        ,inner_weight)                              // gradient bias i & j
        
        
     })     
     
     w = w.map( item => 
       if ( item._1 == gradients(0)._1._1 ) (item._1 , item._2 - gradients(0)._2 * learningRate / Math.sqrt(iter) )
       else 
       if ( item._1 == gradients(0)._1._2 ) (item._1 , item._2 - gradients(0)._3 * learningRate / Math.sqrt(iter) )
       else 
         item
     ) 
    W = sc.broadcast(w.collectAsMap())

     
     b = b.map( item => 
       if ( item._1 == gradients(0)._1._1 ) (item._1 , item._2 - gradients(0)._4 * learningRate / Math.sqrt(iter) )
       else 
       if ( item._1 == gradients(0)._1._2 ) (item._1 , item._2 - gradients(0)._4 * item._2 * learningRate / Math.sqrt(iter) )
       else 
         item
     ) 
    B = sc.broadcast(b.collectAsMap())
*/
    val gradients = counts.map(pair => {
      val inner_cost = weighting(pair._2) * (
          W.value(pair._1._1).dot(W.value(pair._1._2)) +
        B.value(pair._1._1) + B.value(pair._1._2) 
        - Math.log(pair._2)
        )
      
      ((pair._1._1, pair._1._2), // i , j
          W.value(pair._1._2) * inner_cost // gradient i
        , W.value(pair._1._1) * inner_cost // gradient j
        , inner_cost) // gradient bias i & j

    })

    // recalculate w 
    w = gradients.flatMap(item =>
      Set(
        (item._1._1, (item._2 * learningRate / Math.sqrt(iter), 1)),
        (item._1._2, (item._3 * learningRate / Math.sqrt(iter), 1))))
      .reduceByKey((g, c) => (g._1 + g._1, c._2 + c._2))
      .map(item => (item._1, W.value(item._1) - item._2._1.:/(item._2._2.toDouble)))
    W = sc.broadcast(w.collectAsMap())

    // recalculate b 
    b = gradients.flatMap(item =>
      Set(
        (item._1._1, (item._4 * learningRate / Math.sqrt(iter), 1)),
        (item._1._2, (item._4 * learningRate / Math.sqrt(iter), 1))))
      .reduceByKey((g, c) => (g._1 + g._1, c._2 + c._2))
      .map(item => (item._1, B.value(item._1) - item._2._1.:/(item._2._2.toDouble)))
    B = sc.broadcast(b.collectAsMap())

    // calculate cost
  val cost = counts.map(pair => (
    weighting(pair._2) *
    Math.pow(W.value(pair._1._1).dot(W.value(pair._1._2)) +
      B.value(pair._1._1) + B.value(pair._1._1)
      - Math.log(pair._2), 2)))
    .reduce(_ + _)
    log("Iteration " + iter + " cost: " + cost)
    log("W size: " + W.value.size)
    log("B size: " + B.value.size)
  log("W 1 2 3: " + W.value(1) + " ; " + W.value(2) + " ; " + W.value(3) )
  log("B 1 2 3: " + B.value(1) + " ; " + B.value(2) + " ; " + B.value(3) )

  }

  log("End word vectors, saving to " + savedVectorFiles);
  w.saveAsObjectFile(savedVectorFiles)
  b.saveAsObjectFile(savedVectorFiles + "Bias")

  log("============ End Run ========");

  def isConverged(
    previousWeights: DenseVector[Double],
    currentWeights: DenseVector[Double],
    convergenceTol: Double): Boolean = {
    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousWeights - currentWeights)
    solutionVecDiff < convergenceTol * Math.max(norm(currentWeights), 1.0)
  }
}