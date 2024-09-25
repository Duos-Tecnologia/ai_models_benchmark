package com.example.ai_models_benchmark

import com.example.ai_models_benchmark.tflite.Recognition
import java.time.Duration

class DetectionResult(
    val inferenceTime: Duration,
    val recognitions: List<Recognition>
) {
}