package com.example.ai_models_benchmark.tflite

import android.graphics.RectF

public class Recognition(
    val labelId: Int,
    var name: String,
    val score: Float,
    val confidence: Float,
    val location: RectF,
) {

}