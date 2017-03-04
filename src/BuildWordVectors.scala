

import java.io.BufferedWriter
import java.io.FileWriter
import java.io.PrintWriter

import scala.util.Random

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.GloveGradient
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

object BuildWordVectors extends App {

  //val savedRDDfiles = "coocMatrixSmall";
  //val savedVectorFiles = "WordVectorsSmall";
  val savedRDDfiles = "coocMatrix";
  val reloadVectorFiles = "";
  val savedVectorFiles = "WordVectors";

  val logFile = "BuildWordVectors.log" // optional log file 
  val stepSize = 1
  val MAX_ITER = 15 // מספר איטרציות
  val X_MAX = 100 //  ערכו של X_MAX בנוסחה 9 במאמר
  val ALPHA = 0.75 // ערכו של פרמטר Alpha בנוסחה 9 במאמר
  val VECTOR_SIZE = 50 // גודל הווקטור לייצוג המילה  
  val miniBatchFraction = 0.3

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
  val conf = new SparkConf().setAppName("GloVe").setMaster("local")
  .set("spark.driver.maxResultSize", "3g")
  val sc = new SparkContext(conf)
  //raise log level to ERROR - ignoring the INFO messages from spark 
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)
  val gradient = new GloveGradient(VECTOR_SIZE, X_MAX, ALPHA);

  // Read the cooccurrence matrix
  // -----------------------------

  log("Start word vectors, reading from " + savedRDDfiles);

  // restore saved RDD's
  val wordKeys = sc.objectFile[(String, Int)](savedRDDfiles + "Keys")
  val counts = sc.objectFile[((Int, Int), Int)](savedRDDfiles)
  log("Number of cooccurrence observations: " + counts.count())

  val vocabularySize = wordKeys.count().toInt;
  val maxWordId = wordKeys.map(_._2).reduce((a, b) => (if (a > b) a else b))
  log("vocabularySize is " + vocabularySize)
  log("maxWordId is " + maxWordId)

  log("Running SGD, miniBatchFraction is " + miniBatchFraction + ", MAX_ITER = " + MAX_ITER + " stepSize = " + stepSize + " VECTOR_SIZE = " + VECTOR_SIZE)

  // Build the words vectors
  // -----------------------------
  val numRows = maxWordId.toInt + 1;
  
  // start with random weights
  // OR start with known weights from previous optimization
  val initialWeights : Array[Vector] = if (reloadVectorFiles.isEmpty()) {
     Array.fill(numRows)(Vectors.dense(Array.fill(VECTOR_SIZE)(Random.nextDouble)))
  } else {
     sc.objectFile[Vector](reloadVectorFiles).collect()
  }
  val initialBiases: Vector = if (reloadVectorFiles.isEmpty()) {
     Vectors.dense(Array.fill(numRows)(Random.nextDouble))
  } else {
     Vectors.dense(sc.objectFile[Double](reloadVectorFiles + "Bias").collect)
  }
  
  //--------------------------
  //log("Dictionary: \n" + wordKeys.map(pair =>  pair._1 +" => " +  pair._2 + "\n").reduce(_+_) )
  //log("Counts Matrix: \n" + counts.map(pair => "WORDS: " + pair._1._1 +","+pair._1._2+" => " + pair._2 + "\n").reduce(_+_) )
  //log("Initial Weights: " + gradient.VtoS(initialWeights) )
  //log("Initial Biases: (size=" + initialBiases.size +") First 10 items: " + initialBiases.toArray.take(10).mkString(" "))
  val cost0 = counts.map(pair => (
    gradient.singleCost(initialWeights, initialBiases, pair._1._1.toInt, pair._1._2.toInt, pair._2)))
    .reduce(_ + _)
  log("Initial cost: " + cost0)

  //--------------------------------

  // 
  var weights = initialWeights.clone
  var biases = initialBiases.copy

  var converged = false // not implemented - always false
  var iter = 1
  while (!converged && iter <= MAX_ITER) {
    // broadcast weights and biases
    val bcWeights = sc.broadcast(weights)
    val bcBiases = sc.broadcast(biases)

    val (gradientSum, biasVector, batchCounts) = counts.sample(false, miniBatchFraction, 34 + iter)
      // aggregate a gradients matrix, bias vector and count vector
      .treeAggregate((
        Array.fill(numRows)(Vectors.dense(Array.fill(VECTOR_SIZE)(0.0))),
        Vectors.dense(Array.fill(numRows)(0.0)),
        Vectors.dense(Array.fill(numRows)(0.0))))(
        seqOp = (c, v) => {
          // c: (gradients_weights, gradients_bias, count), v: ((wordId,wordId), count)
          gradient.compute(c._1, c._2, c._3, v._1._1, v._1._2, v._2, bcWeights.value, bcBiases.value)
          // returns partitioned values: Array[Vector] , Vector , Vector
        },
        combOp = (c1, c2) => {
          // c: (gradients_weights, gradients_bias, count)
          gradient.aggregate(c1._1, c2._1, c1._2, c2._2, c1._3, c2._3)
          // returns aggregated values in: Array[Vector] , Vector , Vector
        })

    // iteration ended
    // update weights and biases
    if (batchCounts.toArray.sum > 0) {
      val thisIterStepSize = stepSize / Math.sqrt(iter)
      val results = gradient.update(gradientSum, weights, biasVector, biases, batchCounts, thisIterStepSize)
      weights = results._1
      biases = results._2
      //Logger.getRootLogger().error("Updated Weight gradients: " + gradient.VtoS(weights) )
      //Logger.getRootLogger().error("Updated Bias gradients: (size=" + initialBiases.size +") First 10 items: " + biases.toArray.take(10).mkString(" "))
    } else {
      log(s"Iteration ($iter/$MAX_ITER). The size of sampled batch is zero")
    }
    log("Finished iteration " + iter)
    iter += 1 // next iteration
  }

  log("End word vectors, saving to " + savedVectorFiles);

  //--------------------------
  val cost1 = counts.map(pair => (
    gradient.singleCost(weights, biases, pair._1._1.toInt, pair._1._2.toInt, pair._2)))
    .reduce(_ + _)
  //log("Optimal Weights: (size=" + result._1.size+") " + result._1)
  //log("Optimal Weights: " + gradient.VtoS(weights) )
  log("Optimized cost: " + cost1)

  //--------------------------------

  sc.parallelize(weights).saveAsObjectFile(savedVectorFiles)
  sc.parallelize(biases.toArray).saveAsObjectFile(savedVectorFiles + "Bias")

  log("============ End Run ========");

}