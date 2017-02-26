package org.bgu.mdm.ex4

import breeze.linalg._
import breeze.linalg.functions._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.GloveGradient
import org.apache.log4j.{ Level, Logger }

object ReadWordVectors extends App {

  val savedRDDfiles = "coocMatrixSmall";
  val savedVectorFiles = "wordVectorsSmall";

  val test = Array(
    Array("man", "woman", "king"),
    Array("france", "paris", "italy"),
    Array("boy", "man", "girl"),
    Array("strong", "stronger", "clear"))

  // initialize spark 
  val conf = new SparkConf().setAppName("CoClustering").setMaster("local")
  val sc = new SparkContext(conf)
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)

  val resultLoss = sc.objectFile[Double](savedVectorFiles + "Loss")
  val b = new StringBuilder()
  resultLoss.collect().addString(b, ",")
  println("Optimization interations: " + b.toString())

  val weights = Vectors.dense(sc.objectFile[Double](savedVectorFiles).collect())
  println("Word vectors size: " + weights.size)

  val wk = sc.objectFile[(String, Long)](savedRDDfiles + "Keys")
  val wordKeys = wk.collectAsMap()
  println("Word dictionary size: " + wordKeys.size)
  // println("Word dictionary first items: " + wordKeys.map(item => (item._1 +" : " + item._2) ).reduce( _ + " \n" + _)  )

  val gg = new GloveGradient();

  test.foreach(w3 => {

    println("======================")
    println("Question: " + w3(0) + " -> " + w3(1) + " = " + w3(2) + " -> ? ")
    if (wordKeys.contains(w3(0)) && wordKeys.contains(w3(2)) && wordKeys.contains(w3(2))) {
      println("Top 10 answers by cosine similarity:")
      val diff0 = _cosineDistance(gg.word_vector(wordKeys.get(w3(0)).get.toInt, weights),
        gg.word_vector(wordKeys.get(w3(1)).get.toInt, weights))
      val base0 = new DenseVector[Double](gg.word_vector(wordKeys.get(w3(2)).get.toInt, weights).toArray)
      wk.map(item => (Math.abs(diff0 - _cosineDistance(base0, gg.word_vector(item._2.toInt, weights))), item._1))
        .sortByKey(true).take(10).map(item => "     " + item._2 + ": " + item._1).foreach(println)

          println("Top 10 answers by euclidean distance:")
      val diff1 = _euclideanDistance(gg.word_vector(wordKeys.get(w3(0)).get.toInt, weights),
        gg.word_vector(wordKeys.get(w3(1)).get.toInt, weights))
      val base1 = new DenseVector[Double](gg.word_vector(wordKeys.get(w3(2)).get.toInt, weights).toArray)
      wk.map(item => (Math.abs(diff1 - _euclideanDistance(base1, gg.word_vector(item._2.toInt, weights))), item._1))
        .sortByKey(true).take(10).map(item => "     " + item._2 + ": " + item._1).foreach(println)

    } else {
      println("Not all words appear in vocabulary...")
    }
  })

  private def _cosineDistance(a: Vector, b: Vector): Double = {
    cosineDistance(
      new DenseVector[Double](a.toArray),
      new DenseVector[Double](b.toArray))
  }
  private def _cosineDistance(a: DenseVector[Double], b: Vector): Double = {
    cosineDistance(
      a,
      new DenseVector[Double](b.toArray))
  }
  private def _euclideanDistance(a: Vector, b: Vector): Double = {
    euclideanDistance(
      new DenseVector[Double](a.toArray),
      new DenseVector[Double](b.toArray))
  }
  private def _euclideanDistance(a: DenseVector[Double], b: Vector): Double = {
    euclideanDistance(
      a,
      new DenseVector[Double](b.toArray))
  }
}