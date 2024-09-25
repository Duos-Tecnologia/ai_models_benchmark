package com.example.ai_models_benchmark.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import com.example.ai_models_benchmark.AIModel
import com.example.ai_models_benchmark.DetectionResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.InputStream
import java.nio.ByteBuffer
import java.time.Duration
import java.time.LocalDateTime
import java.util.Arrays
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

class TfliteDetector {
    private val DETECT_THRESHOLD = 0.25f
    private val IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f
    private val IOU_THRESHOLD = 0.45f

    private lateinit var model: AIModel
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    private lateinit var heartBitmap: Bitmap

    fun initializeModel(activity: Context, initModel: AIModel) {
        val bitmap: InputStream = activity.assets.open("heart.png")
        heartBitmap = BitmapFactory.decodeStream(bitmap)

        model = initModel

        val tfliteModel: ByteBuffer = FileUtil.loadMappedFile(activity, model.modelFile)

        val options = Interpreter.Options()
        // options.addDelegate(NnApiDelegate())

        interpreter = Interpreter(tfliteModel, options)
        labels = FileUtil.loadLabels(activity, model.labelsFile)
    }

    fun detect(bitmap: Bitmap): DetectionResult {
        val inputHeight = model.inputShape.height
        val inputWidth = model.inputShape.width
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0F, 255F)) // Normaliza para 0-1
            .build()

        var tensorInput = TensorImage(DataType.FLOAT32)

        tensorInput.load(bitmap)
        tensorInput = imageProcessor.process(tensorInput)

        // Obtenha detalhes da saída
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape() // Exemplo: [1, atributos, detecções]
        val numAttributes = outputShape[1]
        val numDetections = outputShape[2]

        val probabilityBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        val timeBeforeInference = LocalDateTime.now()
        interpreter.run(tensorInput.buffer, probabilityBuffer.buffer)
        val inferenceTime = Duration.between(timeBeforeInference, LocalDateTime.now())

        val recognitionArray = probabilityBuffer.floatArray

        // Remodelar o array unidimensional em um array 2D
        val outputData = Array(numAttributes) { FloatArray(numDetections) }
        var index = 0
        for (i in 0 until numAttributes) {
            for (j in 0 until numDetections) {
                outputData[i][j] = recognitionArray[index++]
            }
        }

        val allRecognitions = ArrayList<Recognition>()

        val numClasses = numAttributes - 5 // Supondo que os 5 primeiros são x, y, w, h, confiança

        for (i in 0 until numDetections) {
            val confidence = outputData[4][i]
            if (confidence > DETECT_THRESHOLD) {
                val x = outputData[0][i] * bitmap.width
                val y = outputData[1][i] * bitmap.height
                val w = outputData[2][i] * bitmap.width
                val h = outputData[3][i] * bitmap.height

                val xmin = max(0.0f, x - w / 2.0f)
                val ymin = max(0.0f, y - h / 2.0f)
                val xmax = min(bitmap.width.toFloat(), x + w / 2.0f)
                val ymax = min(bitmap.height.toFloat(), y + h / 2.0f)

                // Obter pontuações das classes
                val classScores = FloatArray(numClasses)
                for (c in 0 until numClasses) {
                    classScores[c] = outputData[5 + c][i]
                }

                // Identificar a classe com maior pontuação
                var labelId = 0
                var maxLabelScore = 0f
                for (j in classScores.indices) {
                    if (classScores[j] > maxLabelScore) {
                        maxLabelScore = classScores[j]
                        labelId = j
                    }
                }

                val recognition = Recognition(
                    labelId,
                    labels[labelId],
                    maxLabelScore,
                    confidence,
                    RectF(xmin, ymin, xmax, ymax)
                )
                allRecognitions.add(recognition)
            }
        }

        val nmsRecognitions = nms(allRecognitions)
        val finalRecognitions = nmsAllClass(nmsRecognitions)

        return DetectionResult(inferenceTime, finalRecognitions)
    }

    private fun nms(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition> {
        val nmsRecognitions = ArrayList<Recognition>()

        val numClasses = labels.size

        // Para cada classe
        for (i in 0 until numClasses) {
            val pq = PriorityQueue<Recognition>(
                Comparator { l, r ->
                    r.confidence.compareTo(l.confidence)
                }
            )

            // Filtrar reconhecimentos da mesma classe
            for (rec in allRecognitions) {
                if (rec.labelId == i && rec.confidence > DETECT_THRESHOLD) {
                    pq.add(rec)
                }
            }

            // Aplicar NMS
            while (pq.isNotEmpty()) {
                val max = pq.poll()
                nmsRecognitions.add(max)

                val iterator = pq.iterator()
                while (iterator.hasNext()) {
                    val detection = iterator.next()
                    if (boxIou(max.location, detection.location) > IOU_THRESHOLD) {
                        iterator.remove()
                    }
                }
            }
        }
        return nmsRecognitions
    }

    private fun nmsAllClass(allRecognitions: ArrayList<Recognition>): ArrayList<Recognition> {
        val nmsRecognitions = ArrayList<Recognition>()

        val pq = PriorityQueue<Recognition>(
            Comparator { l, r ->
                r.confidence.compareTo(l.confidence)
            }
        )

        // Adicionar todos os reconhecimentos acima do limiar
        for (rec in allRecognitions) {
            if (rec.confidence > DETECT_THRESHOLD) {
                pq.add(rec)
            }
        }

        // Aplicar NMS entre classes
        while (pq.isNotEmpty()) {
            val max = pq.poll()
            nmsRecognitions.add(max)

            val iterator = pq.iterator()
            while (iterator.hasNext()) {
                val detection = iterator.next()
                if (boxIou(max.location, detection.location) > IOU_CLASS_DUPLICATED_THRESHOLD) {
                    iterator.remove()
                }
            }
        }
        return nmsRecognitions
    }

    private fun boxIou(a: RectF, b: RectF): Float {
        val intersection = boxIntersection(a, b)
        val union = boxUnion(a, b)
        if (union <= 0) return 0f
        return intersection / union
    }

    private fun boxIntersection(a: RectF, b: RectF): Float {
        val maxLeft = max(a.left, b.left)
        val maxTop = max(a.top, b.top)
        val minRight = min(a.right, b.right)
        val minBottom = min(a.bottom, b.bottom)
        val w = minRight - maxLeft
        val h = minBottom - maxTop

        if (w < 0 || h < 0) return 0f
        return w * h
    }

    private fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        val u = (a.width() * a.height()) + (b.width() * b.height()) - i
        return u
    }
}
