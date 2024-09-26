package com.example.ai_models_benchmark

import android.util.Size
import org.tensorflow.lite.support.metadata.MetadataExtractor

class AIModel(
    val name: String,
    val videoExample: Int,
    val modelFile: String,
    val labelsFile: String,
    val inputShape: Size,
    val outputShape: IntArray,
    val isInt8: Boolean,
    val inputQuantParams: MetadataExtractor.QuantizationParams? = null,
    val outputQuantParams: MetadataExtractor.QuantizationParams? = null,
) {
    override fun toString(): String {
        return this.name // What to display in the Spinner list.
    }

    val lowOutputTensor
        get() = if(this.outputShape[1] > this.outputShape[2]) this.outputShape[2] else this.outputShape[1]

    val higherOutputTensor
        get() = if(this.outputShape[1] > this.outputShape[2]) this.outputShape[1] else this.outputShape[2]

}