//package org.apache.spark.mllib.linalg

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import breeze.linalg.functions._

object ReadWordVectors extends App {

  //val savedRDDfiles = "coocMatrixSmall";
  //val savedVectorFiles = "wordVectorsSmall";
  val savedRDDfiles = "coocMatrix";
  val savedVectorFiles = "wordVectors";
  val X_MAX = 100 //  ערכו של X_MAX בנוסחה 9 במאמר
  val ALPHA = 0.75 // ערכו של פרמטר Alpha בנוסחה 9 במאמר
  val VECTOR_SIZE = 2 // גודל הווקטור לייצוג המילה  

  // test cases
  val test = Array(
    Array("man", "woman", "king"),
    Array("france", "paris", "italy"),
    Array("boy", "man", "girl"),
    Array("strong", "stronger", "clear"))

  // initialize spark 
  val conf = new SparkConf().setAppName("GloVe").setMaster("local")
  val sc = new SparkContext(conf)
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)

  // load word dictionary
  val wordKeysRDD = sc.objectFile[(String, Int)](savedRDDfiles + "Keys")
  val wordKeys = wordKeysRDD.collectAsMap()
  // load saved word vectors and biases
  val weightsRDD = sc.objectFile[Vector](savedVectorFiles)
  val weights = weightsRDD.collect()
  val biases = sc.objectFile[Double](savedVectorFiles + "Bias").collect

  // evaluate test cases, find closest candidates for the 4th word
  test.foreach(w3 => {

    val a = wordKeys(w3(0)).toInt

    println("======================")
    println("Question: " + w3(0) + " -> " + w3(1) + " = " + w3(2) + " -> ? ")
    if (wordKeys.contains(w3(0)) && wordKeys.contains(w3(1)) && wordKeys.contains(w3(2))) {
      println("Biases: " + biases(wordKeys(w3(1)).toInt) + ", " + biases(wordKeys(w3(1)).toInt) + ", " + biases(wordKeys(w3(2)).toInt) )
      val biasDelta = Math.round((biases(wordKeys(w3(0)).toInt) - biases(wordKeys(w3(1)).toInt))*10)
      
      println("Top 10 answers by cosine similarity:")

      val word3Vector = weights(wordKeys(w3(2)).toInt)

      val diff0 = _cosineDistance(weights(wordKeys(w3(0)).toInt), weights(wordKeys(w3(1)).toInt))
      wordKeysRDD
        .filter( item => biasDelta == Math.round((biases(wordKeys(w3(2)).toInt) - biases(item._2))*10) )
        .map(item => ( Math.abs(diff0 - _cosineDistance(word3Vector, weights(item._2)) ), (item._1, item._2) ) )
        .sortByKey(true).take(10).map(item => "     " + item._2._1 + " (" +biases(item._2._2) + "): " + item._1 ).foreach(println)

      println("Top 10 answers by euclidean distance:")
      val diff1 = _euclideanDistance(weights(wordKeys(w3(0)).toInt), weights(wordKeys(w3(1)).toInt))
      wordKeysRDD
        .filter( item => biasDelta == Math.round((biases(wordKeys(w3(2)).toInt) - biases(item._2))*10) )
        .map(item => ( Math.abs(diff1 - _euclideanDistance(word3Vector, weights(item._2)) ), (item._1, item._2) ) )
        .sortByKey(true).take(10).map(item => "     " + item._2._1 + " (" +biases(item._2._2) + "): " + item._1 ).foreach(println)

    } else {
      println("Not all words appear in vocabulary...")
    }
  })

  // breeze distance calculations
  def _cosineDistance(a: Vector, b: Vector): Double = {
    cosineDistance(new breeze.linalg.DenseVector[Double](a.toArray),
      new breeze.linalg.DenseVector[Double](b.toArray))
  }
  def _euclideanDistance(a: Vector, b: Vector): Double = {
    euclideanDistance(new breeze.linalg.DenseVector[Double](a.toArray),
      new breeze.linalg.DenseVector[Double](b.toArray))
  }

}