package org.apache.spark.mllib.linalg

import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.BLAS.dot
import org.apache.spark.mllib.linalg.BLAS.scal
import org.apache.spark.mllib.optimization.Gradient
import org.apache.log4j.{ Level, Logger }

/*
 * parameters:
 * data: (i,j) word ID's
 * label: Xij word cooccurrence count
 * weights: Vector of size: (vocabulary) times ((word vector size) + (1 for bias))
 * 
 * returns:
 * Vector: gradient (0 for all words except i,j)
 * Double: loss
 */
class GloveGradient(vectorSize: Int, xMax: Int, alpha: Double) extends Gradient {
  var cnt = 0;
  // weighting function - equation 9
  def _weighting(x: Double): Double = {
    if (x < xMax)
      return Math.pow(x / xMax, alpha)
    else
      return 1
  }
  private def _w_start(ij: Int, data: Vector): Int = {
    return data(ij).toInt * (vectorSize + 1)
  }
  private def _w_end(ij: Int, data: Vector): Int = {
    return (1 + data(ij).toInt) * (vectorSize + 1) - 1 // last item is -2, not -1, but slice is taking start <= index(x) < end
  }
  private def _b_pos(ij: Int, data: Vector): Int = {
    return (1 + data(ij).toInt) * (vectorSize + 1) - 1
  }
  private def _wi(data: Vector, weights: Vector): Vector = {
    return Vectors.dense(weights.toArray.slice(_w_start(0, data), _w_end(0, data)))
  }

  private def _wj(data: Vector, weights: Vector): Vector = {
    return Vectors.dense(weights.toArray.slice(_w_start(1, data), _w_end(1, data)))
  }

  private def _bi(data: Vector, weights: Vector): Double = {
    return weights.toArray(_b_pos(0, data))
  }

  private def _bj(data: Vector, weights: Vector): Double = {
    return weights.toArray(_b_pos(1, data))
  }

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val wi = _wi(data, weights)
    val wj = _wj(data, weights)
    val bi = _bi(data, weights)
    val bj = _bj(data, weights)
    val weighting = _weighting(label)
    val inner_cost = (dot(wi, wj) + bi + bj - Math.log(label))
    val loss = weighting * inner_cost * inner_cost
    val db = weighting * inner_cost
    var dwi = Vectors.dense(Array.fill[Double](vectorSize)(0))
    axpy(db, wj, dwi);  //dwi = wj * (inner_cost * weighting)
    var dwj = Vectors.dense(Array.fill[Double](vectorSize)(0))
    axpy(db, wi, dwj);  //dwj = wi * (inner_cost * weighting)
    var gradient = Array.fill[Double](weights.size)(0)
    gradient(_b_pos(0, data)) = db
    gradient(_b_pos(1, data)) = db
    dwi.toArray.copyToArray(gradient, _w_start(0, data), vectorSize)
    dwj.toArray.copyToArray(gradient, _w_start(1, data), vectorSize)
    cnt += 1;
    /*Logger.getRootLogger().error("\n--------\ndata: " + data + "\nlabel: " + label +
        "\nweighting: " + weighting +
        "\ninner_cost: " + inner_cost +
        "\ndwi: " + dwi +
        "\ndwj: " + dwj +
        "\nweights: " + weights +
        "\ngradient: "+Vectors.dense(gradient) +
        "\nLoss: "+loss+
        "\nrun number " + cnt)
    */
    (Vectors.dense(gradient), loss)
  }

  override def compute(
    data: Vector,
    label: Double,
    weights: Vector,
    cumGradient: Vector): Double = {
    val (gradient, loss) = compute(data, label, weights)
    axpy(1.toDouble, gradient, cumGradient)
    loss
  }
  
  
  /*
   * 
   * The following functions are just used to debugging   
   * 
   */
  def word_vector(wordId: Int, weights: Vector): Vector = {
    Vectors.dense(weights.toArray.slice(wordId * (vectorSize + 1), (1 + wordId) * (vectorSize + 1) - 2))
  }
  def b(wordId: Int, weights: Vector): Double = {
    weights.toArray((1 + wordId) * (vectorSize + 1) - 1)
  }
  def _dot(a: Vector, b: Vector): Double = {
    dot(a, b)
  }

}

