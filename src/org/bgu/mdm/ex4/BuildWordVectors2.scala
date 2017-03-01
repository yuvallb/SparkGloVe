package org.bgu.mdm.ex4

import java.io.BufferedWriter
import java.io.FileWriter
import java.io.PrintWriter
import org.apache.log4j.{ Level, Logger }

import scala.util.Random

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.GloveGradient
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD

object BuildWordVectors2 extends App {

  val savedRDDfiles = "trivialMatrix";
  val savedVectorFiles = "trivialWordVectors";

  val logFile = "app4.log" // optional log file 
  val stepSize = 0.8
  val MAX_ITER = 50 // מספר איטרציות
  val X_MAX = 100 //  ערכו של X_MAX בנוסחה 9 במאמר
  val ALPHA = 0.75 // ערכו של פרמטר Alpha בנוסחה 9 במאמר
  val VECTOR_SIZE = 2 // גודל הווקטור לייצוג המילה  
  val miniBatchFraction = 1
  
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
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)
  val gradient = new GloveGradient(VECTOR_SIZE, X_MAX, ALPHA);

  // Read the cooccurrence matrix
  // -----------------------------

  log("Start word vectors, reading from " + savedRDDfiles);

  // restore saved RDD's
  val wordKeys = sc.objectFile[(String, Long)](savedRDDfiles + "Keys")
  val counts = sc.objectFile[((Long, Long), Int)](savedRDDfiles)
  log("number of cooccurrence observations: " + counts.count())

  val vocabularySize = wordKeys.count().toInt;
  val maxWordId = wordKeys.map( _._2).reduce( (a,b) => ( if (a>b) a else b ) )
  log("vocabularySize is " + vocabularySize)
  log("maxWordId is " + maxWordId)

    log("Running SGD, miniBatchFraction is " + miniBatchFraction +", MAX_ITER = "+MAX_ITER +" stepSize = "+stepSize+" VECTOR_SIZE = " + VECTOR_SIZE)

  // convert counts to matrix format from RDD of wordId,wordId,count
  // to RDD of count, vector of the two word id's
  val wordsMatrix: RDD[scala.Tuple2[Double, Vector]] = counts.map(item => (item._2, Vectors.dense(item._1._1, item._1._2)));

  // Build the words vectors
  // -----------------------------
  val updater = new SimpleUpdater()
  val initialWeights: Vector = //Vectors.dense(Array.fill((maxWordId.toInt+1) * (VECTOR_SIZE+1) )(Random.nextDouble))
  Vectors.dense(Array[Double](0.9926451880336998,0.5558561744724114,0.8477777131154619,0.30738839693971953,0.5083874253499047,0.4215912357649044,0.975210113077309,0.4385911050797886,0.37714301729644073,0.9921110705763369,0.2658584716956168,0.5544452922438219,0.9315590900061462,0.6173980841943681,0.4084576812727593,0.5832950254051409,0.16571718254639112,0.5009251826799306,0.3839479272493115,0.9052708295041679,0.16865705334412295));

  //--------------------------
  log("Dictionary: \n" + wordKeys.map(pair =>  pair._1 +" => " +  pair._2 + "\n").reduce(_+_) )
  log("Counts Matrix: \n" + counts.map(pair => "WORDS: " + pair._1._1 +","+pair._1._2+" => " + pair._2 + "\n").reduce(_+_) )
  log("Initial Weights: (size=" + initialWeights.size+") " + initialWeights)
    val cost0 = counts.map(pair => (
    gradient._weighting(pair._2) *
    Math.pow(gradient._dot(gradient.word_vector(pair._1._1.toInt, initialWeights),gradient.word_vector(pair._1._2.toInt, initialWeights)) +
      gradient.b(pair._1._1.toInt,initialWeights) + gradient.b(pair._1._1.toInt,initialWeights)
      - Math.log(pair._2), 2)))
    .reduce(_ + _)  
  log("Initial cost: " + cost0)
  
  //--------------------------------
  
  val result = GradientDescent.runMiniBatchSGD(wordsMatrix, gradient, updater, stepSize, MAX_ITER, 0, miniBatchFraction, initialWeights)

  log("End word vectors, saving to " + savedVectorFiles);

      //--------------------------
    val cost1 = counts.map(pair => (
    gradient._weighting(pair._2) *
    Math.pow(gradient._dot(gradient.word_vector(pair._1._1.toInt, result._1),gradient.word_vector(pair._1._2.toInt, result._1)) +
      gradient.b(pair._1._1.toInt,result._1) + gradient.b(pair._1._1.toInt,result._1)
      - Math.log(pair._2), 2)))
    .reduce(_ + _)
  log("Optimal Weights: (size=" + result._1.size+") " + result._1)
  log("Optimized cost: " + cost1)
  
  //--------------------------------

  sc.parallelize(result._1.toArray).saveAsObjectFile(savedVectorFiles)
  sc.parallelize(result._2).saveAsObjectFile(savedVectorFiles + "Loss")

  val b = new StringBuilder()
  result._2.addString(b, ",")
  println("Optimization interations: " + b.toString())


  log("============ End Run ========");

}