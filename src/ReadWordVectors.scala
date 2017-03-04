//package org.apache.spark.mllib.linalg

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD._

import breeze.linalg.functions.cosineDistance

object ReadWordVectors extends App {

  //val savedRDDfiles = "coocMatrixSmall";
  //val savedVectorFiles = "wordVectorsSmall";
  val savedRDDfiles = "coocMatrix";
  val savedVectorFiles = "wordVectors";
  
  // test cases
  val test = Array(
    Array("man", "woman", "king", "queen"),
    Array("france", "paris", "italy", "rome"),
    Array("boy", "man", "girl", "woman"),
    Array("strong", "stronger", "clear", "clearer"))

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

  // evaluate test cases, find closest candidates for the 4th word
  test.foreach(w3 => {

    val a = wordKeys(w3(0)).toInt

    println("========= " +savedVectorFiles+ " =============")
    println("Question: " + w3(0) + " -> " + w3(1) + " = " + w3(2) + " -> ? ")
    if (wordKeys.contains(w3(0)) && wordKeys.contains(w3(1)) && wordKeys.contains(w3(2))) {

      val word3Vector = weights(wordKeys(w3(2)).toInt)

      val realDist0 = _cosineDistance(weights(wordKeys(w3(2)).toInt), weights(wordKeys(w3(3)).toInt))
      val diff0 = _cosineDistance(weights(wordKeys(w3(0)).toInt), weights(wordKeys(w3(1)).toInt))
      println("Correct answer: " + w3(3) + ": " + (diff0 - realDist0))
      val byCos = wordKeysRDD
        .map(item => (Math.abs(diff0 - _cosineDistance(word3Vector, weights(item._2))), (item._1, item._2)))
        .sortByKey(true)
      byCos.zipWithIndex()
        .filter(item => item._1._2._1 == w3(3))
        .map(item => "     " + item._1._2._1 + ": place " + item._2).foreach(println)
      println("Top 10 answers by cosine similarity:")
      byCos.take(10)
        .map(item => "     " + item._2._1 + ": " + item._1).foreach(println)

    } else {
      println("Not all words appear in vocabulary...")
    }
  })

  // breeze distance calculations
  def _cosineDistance(a: Vector, b: Vector): Double = {
    cosineDistance(new breeze.linalg.DenseVector[Double](a.toArray),
      new breeze.linalg.DenseVector[Double](b.toArray))
  }
}