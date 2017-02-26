package org.bgu.mdm.ex4

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.GloveGradient

object ReadWordVectors extends App {
  val savedRDDfiles = "coocMatrix";
  val savedVectorFiles = "wordVectors";

  // initialize spark 
  val conf = new SparkConf().setAppName("CoClustering").setMaster("local")
  val sc = new SparkContext(conf)

  val resultLoss = sc.objectFile[Double](savedVectorFiles + "Loss")
  val b = new StringBuilder()
  resultLoss.collect().addString(b, ",")
  println("Optimization interations: " + b.toString())

  val weights = Vectors.dense(sc.objectFile[Double](savedVectorFiles).collect())
  println("Word vectors size: " + weights.size)

  val wordKeys = sc.objectFile[(String, Long)](savedRDDfiles + "Keys").collectAsMap()
  println("Word dictionary size: " + wordKeys.size)
  //println("Word dictionary first items: " + wordKeys.take(100).map(item => (item._1) ).reduce(_ +", "+ _ ) )
  
  val gg = new GloveGradient();
  println("Hard: (" + wordKeys.get("hard").get.toString + ") Vector: " + gg.word_vector(wordKeys.get("hard").get.toInt, weights))
  println("Easy: (" + wordKeys.get("easy").get.toString + ") Vector: " + gg.word_vector(wordKeys.get("easy").get.toInt, weights))
  println("War: (" + wordKeys.get("war").get.toString + ") Vector: " + gg.word_vector(wordKeys.get("war").get.toInt, weights))
  println("Peace: (" + wordKeys.get("peace").get.toString + ") Vector: " + gg.word_vector(wordKeys.get("peace").get.toInt, weights))
  println("Nice: (" + wordKeys.get("nice").get.toString + ") Vector: " + gg.word_vector(wordKeys.get("nice").get.toInt, weights))
  println("Guy: (" + wordKeys.get("guy").get.toString + ") Vector: " + gg.word_vector(wordKeys.get("guy").get.toInt, weights))
}