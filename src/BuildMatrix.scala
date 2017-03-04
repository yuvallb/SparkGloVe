

import java.io.BufferedWriter
import java.io.FileWriter
import java.io.PrintWriter

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.rdd.RDDFunctions.fromRDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

object BuildMatrix extends App {

  // train files 
  //val trainFile = "COCA/small/w_mag*.txt" 
  val trainFile = "COCA/text/*.txt" 
  val savedRDDfiles = "coocMatrix";

  val logFile = "BuildMatrix.log" // optional log file 
  val VOCAB_MIN_COUNT = 5 // מספר הופעות מינימלי של מלה כדי להיכלל בחישוב
  val WINDOW_SIZE = 15 //  גודל החלון לחישוב co-occurrences

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
  val sc = new SparkContext(conf)

  // Build the cooccurrence matrix
  // -----------------------------

  log("Start cooccurrence matrix, reading from " + trainFile);
  val textFile = sc.textFile(trainFile)
  val all_words = textFile.flatMap(line => line.split(" ")).map(word => word.toLowerCase())

  val wordKeys = all_words.map(word => (word, 1))
    .reduceByKey(_ + _).filter(_._2 >= VOCAB_MIN_COUNT)
    .filter(_._1.length() > 1).filter(_._1.matches("[a-z]+"))
    .zipWithUniqueId().map(wordKeyCount => (wordKeyCount._1._1, wordKeyCount._2.toInt))

  val wordKeysMap = sc.broadcast(wordKeys.collectAsMap());

  val counts = all_words.sliding(WINDOW_SIZE)
    .flatMap(
      wordsWindow => wordsWindow.takeRight(WINDOW_SIZE - 1).map((_, wordsWindow(0))))
    .filter(pair => !pair._1.equals(pair._2))
    .filter(pair => wordKeysMap.value.contains(pair._1) && wordKeysMap.value.contains(pair._2))
    .map(pair => (wordKeysMap.value(pair._1), wordKeysMap.value(pair._2)))
    .map(pair => if (pair._1 > pair._2) ((pair._1, pair._2), 1) else ((pair._2, pair._1), 1))
    .reduceByKey(_ + _)

  wordKeys.saveAsObjectFile(savedRDDfiles + "Keys")
  counts.saveAsObjectFile(savedRDDfiles)

  log("End cooccurrence matrix, saving to " + savedRDDfiles);

  log("============ End Run ========");

}