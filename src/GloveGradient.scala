package org.apache.spark.mllib.linalg

import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.linalg.BLAS.{ axpy, dot, scal }
import org.apache.spark.mllib.util.MLUtils

/*
 * parameters:
 * data: (i,j) word ID's
 * label: Xij word coocurence count
 * weights: Vector of size vocabulary X word vector size
 * 
 * returns:
 * Vector: gradient (0 for all words except i,j)
 * Double: loss
 */
class GloveGradient extends Gradient {

  val X_MAX = 100 //  ערכו של X_MAX בנוסחה 9 במאמר
  val ALPHA = 0.75 // ערכו של פרמטר Alpha בנוסחה 9 במאמר
  val VECTOR_SIZE = 50 // גודל הווקטור לייצוג המילה

  // weighting function - equation 9
  private def _weighting(x: Double): Double = {
    return if (x < X_MAX)
      Math.pow(x / X_MAX, ALPHA)
    else
      1
  }
  private def _w_start(ij: Int, data: Vector): Int = {
    data(ij).toInt * (VECTOR_SIZE+1)
  }
  private def _w_end(ij: Int, data: Vector): Int = {
    (1+data(ij).toInt) * (VECTOR_SIZE+1) - 2
  }
  private def _b_pos(ij: Int, data:Vector): Int = {
    (1+data(ij).toInt) * (VECTOR_SIZE+1) - 1
  }
  private def _wi(data: Vector, weights: Vector): Vector = {
    Vectors.dense(weights.toArray.slice(_w_start(0, data), _w_end(0,data)))
  }

  private def _wj(data: Vector, weights: Vector): Vector = {
    Vectors.dense(weights.toArray.slice(_w_start(1, data), _w_end(1,data)))
  }
  
  private def _bi(data: Vector, weights: Vector): Double = {
    weights.toArray(_b_pos(0, data))
  }

  private def _bj(data: Vector, weights: Vector): Double = {
    weights.toArray(_b_pos(1,data))
  }

  def word_vector(wordId: Int, weights: Vector): Vector = {
    Vectors.dense(weights.toArray.slice(wordId  * (VECTOR_SIZE+1), (1+wordId) * (VECTOR_SIZE+1) - 2 ))
  }
  
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val wi = _wi(data,weights)
    val wj = _wj(data,weights)
    val bi = _bi(data,weights)
    val bj = _bj(data,weights)
    val weighting = _weighting(label)
    val inner_cost = ( dot(wi,wj) + bi + bj - Math.log(label) )
    val loss = weighting * inner_cost * inner_cost
    var dwi = wj.copy
    scal(weighting , dwi) ; scal(inner_cost , dwi) //dwi = weighting * wj + inner_cost
    var dwj = wi.copy
    scal(weighting , dwj) ; scal(inner_cost , dwj) //dwj = weighting * wi + inner_cost
    val db = weighting * inner_cost
    var gradient = Array.fill[Double](weights.size)(0)
    gradient(_b_pos(0,data)) = db
    gradient(_b_pos(1,data)) = db
    dwi.toArray.copyToArray(gradient, _w_start(0,data) , VECTOR_SIZE )
    dwj.toArray.copyToArray(gradient, _w_start(1,data) , VECTOR_SIZE )
    (Vectors.dense(gradient), loss)
  }

  override def compute(
    data: Vector,
    label: Double,
    weights: Vector,
    cumGradient: Vector): Double = {
    val (gradient,loss) = compute(data , label , weights)
    axpy( 1.toDouble , gradient , cumGradient)
    loss
  }
}

