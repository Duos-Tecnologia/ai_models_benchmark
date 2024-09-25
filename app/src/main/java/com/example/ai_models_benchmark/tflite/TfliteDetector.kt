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
        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0F, 255F)).build()

        var tensorInput: TensorImage = TensorImage(DataType.FLOAT32)

        if(model.higherOutputTensor == model.outputShape[1]) {
            tensorInput.load(bitmap)
        }
        else{
            tensorInput.load(heartBitmap)

        }
        tensorInput = imageProcessor.process(tensorInput)

        val probabilityBuffer: TensorBuffer = TensorBuffer.createFixedSize(model.outputShape, DataType.FLOAT32)

        val timeBeforeInference = LocalDateTime.now()
        interpreter.run(tensorInput.buffer, probabilityBuffer.buffer)
        val inferenceTime = Duration.between(timeBeforeInference, LocalDateTime.now())

        val recognitionArray = probabilityBuffer.floatArray

        val allRecognitions = ArrayList<Recognition>()

        //if(model.higherOutputTensor == model.outputShape[1]){
            for (i in 0 until model.higherOutputTensor) {
                val gridStride: Int = i * model.lowOutputTensor
                val confidence = recognitionArray[4 + gridStride]
                if(confidence > 0.6){
                    val x: Float = recognitionArray[0 + gridStride] * bitmap.width
                    val y: Float = recognitionArray[1 + gridStride] * bitmap.height
                    val w: Float = recognitionArray[2 + gridStride] * bitmap.width
                    val h: Float = recognitionArray[3 + gridStride] * bitmap.height
                    val xmin = max(0.0, x - w / 2.0).toInt()
                    val ymin = max(0.0, y - h / 2.0).toInt()
                    val xmax = min(bitmap.width.toDouble(), x + w / 2.0).toInt()
                    val ymax = min(bitmap.height.toDouble(), y + h / 2.0).toInt()

                    val classScores = Arrays.copyOfRange(
                        recognitionArray,
                        5 + gridStride,
                        model.lowOutputTensor + gridStride
                    )

                    var labelId = 0
                    var maxLabelScores = 0f
                    for (j in classScores.indices) {
                        if (classScores[j] > maxLabelScores) {
                            maxLabelScores = classScores[j]
                            labelId = j
                        }
                    }


                    val r = Recognition(
                        labelId,
                        "",
                        maxLabelScores,
                        confidence,
                        RectF(xmin.toFloat(), ymin.toFloat(), xmax.toFloat(), ymax.toFloat())
                    )
                    allRecognitions.add(
                        r
                    )
                }

            }
      //  }else {

//            for (i in 0 until model.higherOutputTensor) {
//                val confidence = recognitionArray[4 + (i * model.lowOutputTensor)]
//                if (confidence > 0.6) {
//                    val x: Float = recognitionArray[0 + (i * model.lowOutputTensor)] * bitmap.width
//                    val y: Float = recognitionArray[1 + (i * model.lowOutputTensor)] * bitmap.height
//                    val w: Float = recognitionArray[2 + (i * model.lowOutputTensor)] * bitmap.width
//                    val h: Float = recognitionArray[3 + (i * model.lowOutputTensor)] * bitmap.height
//                    val xmin = max(0.0, x - w / 2.0).toInt()
//                    val ymin = max(0.0, y - h / 2.0).toInt()
//                    val xmax = min(bitmap.width.toDouble(), x + w / 2.0).toInt()
//                    val ymax = min(bitmap.height.toDouble(), y + h / 2.0).toInt()
//
//                    val classScores = Arrays.copyOfRange(
//                        recognitionArray,
//                        5 + (i * model.lowOutputTensor),
//                        model.lowOutputTensor + (i * model.lowOutputTensor)
//                    )
//
//                    var labelId = 0
//                    var maxLabelScores = 0f
//                    for (j in classScores.indices) {
//                        if (classScores[j] > maxLabelScores) {
//                            maxLabelScores = classScores[j]
//                            labelId = j
//                        }
//                    }
//
//
//                    val r = Recognition(
//                        labelId,
//                        "",
//                        maxLabelScores,
//                        confidence,
//                        RectF(xmin.toFloat(), ymin.toFloat(), xmax.toFloat(), ymax.toFloat())
//                    )
//                    if(r.labelId ==2){
//
//                        allRecognitions.add(
//                            r
//                        )
//                    }
//                }
//            }
       // }


//        for (i in 0 until model.outputShape[1]) {
//            val gridStride: Int = i * model.outputShape[2]
//            // 由于yolov5作者在导出tflite的时候对输出除以了image size, 所以这里需要乘回去
//            val x: Float = recognitionArray[0 + gridStride] * bitmap.width
//            val y: Float = recognitionArray[1 + gridStride] * bitmap.height
//            val w: Float = recognitionArray[2 + gridStride] * bitmap.width
//            val h: Float = recognitionArray[3 + gridStride] * bitmap.height
//            val xmin = max(0.0, x - w / 2.0).toInt()
//            val ymin = max(0.0, y - h / 2.0).toInt()
//            val xmax = min(bitmap.width.toDouble(), x + w / 2.0).toInt()
//            val ymax = min(bitmap.height.toDouble(), y + h / 2.0).toInt()
//            val confidence = recognitionArray[4 + gridStride]
//            val classScores = Arrays.copyOfRange(
//                recognitionArray,
//                5 + gridStride,
//                model.outputShape[2] + gridStride
//            )
//            //            if(i % 1000 == 0){
////                Log.i("tfliteSupport","x,y,w,h,conf:"+x+","+y+","+w+","+h+","+confidence);
////            }
//            var labelId = 0
//            var maxLabelScores = 0f
//            for (j in classScores.indices) {
//                if (classScores[j] > maxLabelScores) {
//                    maxLabelScores = classScores[j]
//                    labelId = j
//                }
//            }
//
//
//            val r = Recognition(
//                labelId,
//                "",
//                maxLabelScores,
//                confidence,
//                RectF(xmin.toFloat(), ymin.toFloat(), xmax.toFloat(), ymax.toFloat())
//            )
//            allRecognitions.add(
//                r
//            )
//        }

        val nmsRecognitions:ArrayList<Recognition> = nms(allRecognitions)

        // 第二次非极大抑制, 过滤那些同个目标识别到2个以上目标边框为不同类别的
        val nmsFilterBoxDuplicationRecognitions: java.util.ArrayList<Recognition> =
            nmsAllClass(nmsRecognitions)


        // 更新label信息
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