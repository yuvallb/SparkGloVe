package org.apache.spark.mllib.linalg

import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.linalg.BLAS.scal
import org.apache.spark.mllib.optimization.Gradient
import org.apache.log4j.{ Level, Logger }

class GloveGradient3(vectorSize: Int, xMax: Int, alpha: Double)  {
  
  // iteration count for debugging
  var cnt = 0;
  
  // weighting function - equation 9
  def _weighting(x: Double): Double = {
    if (x < xMax)
      return Math.pow(x / xMax, alpha)
    else
      return 1
  }
  
  // compute gradients of a single data point
  def compute(gradientsW: Array[Vector], gradientsB: Vector, counts: Vector, wordI: Int, wordJ: Int, Xij: Int, weights: Array[Vector], bias: Vector): (Array[Vector], Vector, Vector) = {
    val wi = weights(wordI)
    val wj = weights(wordJ)
    val weighting = _weighting(Xij)
    val inner_cost = (dot(wi , wj) + bias(wordI) + bias(wordJ) - Math.log(Xij))
    val loss = weighting * inner_cost * inner_cost
    val db = weighting * inner_cost
    var dwi = Vectors.dense(Array.fill[Double](vectorSize)(0))
    axpy(db, wj, dwi);  //dwi = wj * (inner_cost * weighting)
    var dwj = Vectors.dense(Array.fill[Double](vectorSize)(0))
    axpy(db, wi, dwj);  //dwj = wi * (inner_cost * weighting)
    axpy( 1 , dwi, gradientsW(wordI))
    axpy( 1 , dwj, gradientsW(wordJ))
    axpy( 1, Vectors.sparse(counts.size, Array(wordI,wordJ), Array(1,1)) , counts)
    axpy( 1, Vectors.sparse(counts.size, Array(wordI,wordJ), Array(db,db)) , bias)  
  /*
    cnt += 1;
    Logger.getRootLogger().error("\n--------\ndata: " + data + "\nlabel: " + label +
        "\nweighting: " + weighting +
        "\ninner_cost: " + inner_cost +
        "\ndwi: " + dwi +
        "\ndwj: " + dwj +
        "\ndb: " + db +
        "\nweights: " + weights +
        "\ngradient: "+Vectors.dense(gradient) +
        "\nLoss: "+loss+
        "\nrun number " + cnt)
    
    */
    (gradientsW, gradientsB, counts)
  }

  // aggregate two gradients
  def aggregate(
      gradientsW1: Array[Vector], gradientsW2: Array[Vector], 
      gradientsB1: Vector, gradientsB2: Vector, 
      counts1: Vector, counts2: Vector): (Array[Vector], Vector, Vector) = {
    val gradsW = gradientsW1.clone
    for( i <- 0 to gradsW.size - 1 )  {
      axpy(1, gradientsW2(i), gradsW(i))
    }
    gradientsW2.foreach { x => ??? }
    val gradsB = gradientsB1.copy
    axpy(1, gradientsB2, gradsB)
    val counts = counts1.copy
    axpy(1 , counts2 , counts)
    (gradsW, gradsB, counts)
  }
  
  // helper - calculate cost of a single data point
   def singleCost(w: Array[Vector], b: Vector, wordI: Int, wordJ: Int, Xij: Int): Double = {
    _weighting(Xij) *
    Math.pow(dot(w(wordI), w(wordJ)) +
      b(wordI) + b(wordJ)
      - Math.log(Xij), 2)
    }

}

