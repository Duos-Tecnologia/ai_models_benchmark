package com.example.ai_models_benchmark

import android.util.Size

class AIModel(
    val name: String,
    val videoExample: Int,
    val modelFile: String,
    val labelsFile: String,
    val inputShape: Size,
    val outputShape: IntArray,
) {
    override fun toString(): String {
        return this.name // What to display in the Spinner list.
    }

    val lowOutputTensor
        get() = if(this.outputShape[1] > this.outputShape[2]) this.outputShape[2] else this.outputShape[1]

    val higherOutputTensor
        get() = if(this.outputShape[1] > this.outputShape[2]) this.outputShape[1] else this.outputShape[2]

}