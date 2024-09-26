package com.example.ai_models_benchmark.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Log
import com.example.ai_models_benchmark.AIModel
import com.example.ai_models_benchmark.DetectionResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.DequantizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.common.ops.QuantizeOp
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


public class TfliteDetector() {
    private val DETECT_THRESHOLD = 0.25f
    private val IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f
    private val IOU_THRESHOLD = 0.45f

    private lateinit var model: AIModel
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    private lateinit var heartBitmap: Bitmap

    fun initializeModel(activity: Context, initModel: AIModel){
        val bitmap: InputStream =  activity.assets.open("heart.png")
        heartBitmap=BitmapFactory.decodeStream(bitmap);

        model = initModel

        val tfliteModel: ByteBuffer = FileUtil.loadMappedFile(activity, model.modelFile)

        val options: Interpreter.Options = Interpreter.Options()
        //options.addDelegate(NnApiDelegate())

        interpreter = Interpreter(tfliteModel, options)
        labels = FileUtil.loadLabels(activity, model.labelsFile)
    }

    fun detect(bitmap: Bitmap): DetectionResult {
        val inputHeight = model.inputShape.height
        val inputWidth = model.inputShape.width

        val imageProcessor: ImageProcessor
        var tensorInput: TensorImage

        if(model.isInt8){
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0F, 255F))
                .add(QuantizeOp(model.inputQuantParams!!.zeroPoint.toFloat(), model.inputQuantParams!!.scale))
                .add(CastOp(DataType.UINT8)).build()
            tensorInput = TensorImage(DataType.UINT8)
        }else{
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0F, 255F)).build()
            tensorInput = TensorImage(DataType.FLOAT32)
        }

        tensorInput.load(bitmap)

        tensorInput = imageProcessor.process(tensorInput)

        var probabilityBuffer: TensorBuffer

        if(model.isInt8){
            probabilityBuffer = TensorBuffer.createFixedSize(model.outputShape, DataType.UINT8)
        }else{
            probabilityBuffer = TensorBuffer.createFixedSize(model.outputShape, DataType.FLOAT32)
        }

        val timeBeforeInference = LocalDateTime.now()
        interpreter.run(tensorInput.buffer, probabilityBuffer.buffer)
        val inferenceTime = Duration.between(timeBeforeInference, LocalDateTime.now())

        if(model.isInt8){
            val tensorProcessor: TensorProcessor = TensorProcessor.Builder()
                .add(DequantizeOp(model.outputQuantParams!!.zeroPoint.toFloat(), model.outputQuantParams!!.scale)).build()
            probabilityBuffer = tensorProcessor.process(probabilityBuffer)
        }

        val recognitionArray = probabilityBuffer.floatArray
        val allRecognitions = ArrayList<Recognition>()


        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape() // Exemplo: [1, atributos, detecções]
        val numAttributes = model.lowOutputTensor
        //val numAttributes = outputShape[1]
        val numDetections = model.higherOutputTensor
        //val numDetections = outputShape[2]
        val isNumDetectionsInIndexOne = numDetections == outputShape[1]

        val outputData = Array(numAttributes) { FloatArray(numDetections) }
        var index = 0
        if(isNumDetectionsInIndexOne){
            for (i in 0 until numDetections) {
                for (j in 0 until numAttributes) {
                    outputData[j][i] = recognitionArray[index++]
                }
            }
        }else{
            for (i in 0 until numAttributes) {
                for (j in 0 until numDetections) {
                    outputData[i][j] = recognitionArray[index++]
                }
            }
        }

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


        val nmsRecognitions:ArrayList<Recognition> = nms(allRecognitions)

        val nmsFilterBoxDuplicationRecognitions: java.util.ArrayList<Recognition> =
            nmsAllClass(nmsRecognitions)


        for (recognition in nmsFilterBoxDuplicationRecognitions) {
            val labelId: Int = recognition.labelId
            val labelName: String = labels[labelId]
            recognition.name = labelName
        }
        return DetectionResult(inferenceTime,nmsFilterBoxDuplicationRecognitions)
    }

    private fun nms(allRecognitions: java.util.ArrayList<Recognition>): java.util.ArrayList<Recognition> {
        val nmsRecognitions = java.util.ArrayList<Recognition>()

        // 遍历每个类别, 在每个类别下做nms
        for (i in 0 until model.lowOutputTensor - 5) {
            // 这里为每个类别做一个队列, 把labelScore高的排前面
            val pq =
                PriorityQueue<Recognition>(
                    model.higherOutputTensor,
                    object : Comparator<Recognition?> {
                        override fun compare(l: Recognition?, r: Recognition?): Int {
                            if(l == null || r == null){
                                return  -1
                            }
                            // Intentionally reversed to put high confidence at the head of the queue.
                            return r.confidence.compareTo(l.confidence)
                        }
                    })

            // 相同类别的过滤出来, 且obj要大于设定的阈值
            for (j in allRecognitions.indices) {
//                if (allRecognitions.get(j).getLabelId() == i) {
                if (allRecognitions[j].labelId == i && allRecognitions[j].confidence > DETECT_THRESHOLD) {
                    pq.add(allRecognitions[j])
                    //                    Log.i("tfliteSupport", allRecognitions.get(j).toString());
                }
            }

            // nms循环遍历
            while (pq.size > 0) {
                // 概率最大的先拿出来
                val a = arrayOfNulls<Recognition>(pq.size)
                val detections: Array<Recognition> = pq.toArray(a)
                val max = detections[0]
                nmsRecognitions.add(max)
                pq.clear()

                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    if (boxIou(max.location, detection.location) < IOU_THRESHOLD) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsRecognitions
    }

    private fun nmsAllClass(allRecognitions: java.util.ArrayList<Recognition>): java.util.ArrayList<Recognition> {
        val nmsRecognitions = java.util.ArrayList<Recognition>()

        val pq =
            PriorityQueue<Recognition>(
                100,
                object : Comparator<Recognition?> {
                    override fun compare(l: Recognition?, r: Recognition?): Int {
                        if(l == null || r == null){
                            return  -1
                        }
                        // Intentionally reversed to put high confidence at the head of the queue.
                        return r.confidence.compareTo(l.confidence)
                    }
                })

        // 相同类别的过滤出来, 且obj要大于设定的阈值
        for (j in allRecognitions.indices) {
            if (allRecognitions[j].confidence > DETECT_THRESHOLD) {
                pq.add(allRecognitions[j])
            }
        }

        while (pq.size > 0) {
            // 概率最大的先拿出来
            val a = arrayOfNulls<Recognition>(pq.size)
            val detections: Array<Recognition> = pq.toArray(a)
            val max = detections[0]
            nmsRecognitions.add(max)
            pq.clear()

            for (k in 1 until detections.size) {
                val detection = detections[k]
                if (boxIou(
                        max.location,
                        detection.location
                    ) < IOU_CLASS_DUPLICATED_THRESHOLD
                ) {
                    pq.add(detection)
                }
            }
        }
        return nmsRecognitions
    }


    protected fun boxIou(a: RectF, b: RectF): Float {
        val intersection = boxIntersection(a, b)
        val union = boxUnion(a, b)
        if (union <= 0) return 1F
        return intersection / union
    }

    protected fun boxIntersection(a: RectF, b: RectF): Float {
        val maxLeft = if (a.left > b.left) a.left else b.left
        val maxTop = if (a.top > b.top) a.top else b.top
        val minRight = if (a.right < b.right) a.right else b.right
        val minBottom = if (a.bottom < b.bottom) a.bottom else b.bottom
        val w = minRight - maxLeft
        val h = minBottom - maxTop

        if (w < 0 || h < 0) return 0F
        val area = w * h
        return area
    }

    protected fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        val u =
            (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
        return u
    }
}