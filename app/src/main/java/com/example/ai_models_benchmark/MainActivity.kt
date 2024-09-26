package com.example.ai_models_benchmark

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Resources
import android.content.res.Resources.getSystem
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RectF
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.MediaPlayer
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.AdapterView
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.VideoView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.ai_models_benchmark.tflite.TfliteDetector
import com.example.androidscreenstreamer.ImageStreamService
import org.tensorflow.lite.support.metadata.MetadataExtractor
import java.time.Duration
import java.time.LocalDateTime


class MainActivity : AppCompatActivity() {
    companion object {
        private const val NOTIFICATION_PERMISSION_CODE = 0
        private const val SCREEN_CAPTURE_PERMISSION_CODE = 1
        private const val MILLIS_TO_CALCULATE_FPS = 2000L
    }

    private lateinit var tfliteDetector: TfliteDetector

    private lateinit var selectedModel: AIModel

    private lateinit var videoView: VideoView
    private lateinit var imageView: ImageView
    private lateinit var spinner: Spinner
    private lateinit var fpsText: TextView
    private lateinit var inferenceText: TextView
    private lateinit var totalTimeText: TextView
    private lateinit var infMeanText: TextView

    private var boxPaint: Paint = Paint()
    private var textPain: Paint = Paint()


    private val models : List<AIModel> = listOf(
        AIModel(
            name = "yolo_v5",
            videoExample = R.raw.cat,
            modelFile = "yolov5s-fp16.tflite",
            labelsFile = "coco_label.txt",
            inputShape = Size(320, 320),
            outputShape = intArrayOf(1,6300,85),
            isInt8 = false,
            ),
        AIModel(
            name = "train 37 fp16",
            videoExample = R.raw.a4c,
            modelFile = "train_37_fp16.tflite",
            labelsFile = "heart_label.txt",
            inputShape = Size(640, 640),
            outputShape = intArrayOf(1,8,8400),
            isInt8 = false,
        ),
        AIModel(
            name = "train 37 fp32",
            videoExample = R.raw.a4c,
            modelFile = "train_37_fp32.tflite",
            labelsFile = "heart_label.txt",
            inputShape = Size(640, 640),
            outputShape = intArrayOf(1,8,8400),
            isInt8 = false,
        ),
        AIModel(
            name = "train 38 fp16",
            videoExample = R.raw.a4c,
            modelFile = "train_38_fp16.tflite",
            labelsFile = "heart_label.txt",
            inputShape = Size(256, 256),
            outputShape = intArrayOf(1,8,1344),
            isInt8 = false,
        ),
        AIModel(
            name = "train 38 fp32",
            videoExample = R.raw.a4c,
            modelFile = "train_38_fp32.tflite",
            labelsFile = "heart_label.txt",
            inputShape = Size(256, 256),
            outputShape = intArrayOf(1,8,1344),
            isInt8 = false,
        ),
        AIModel(
            name = "train 38 int8",
            videoExample = R.raw.a4c,
            modelFile = "train_38_fp8.tflite",
            labelsFile = "heart_label.txt",
            inputShape = Size(256, 256),
            outputShape = intArrayOf(1,8,1344),
            isInt8 = true,
            inputQuantParams = MetadataExtractor.QuantizationParams(0.003921568859368563f, 128),
            outputQuantParams = MetadataExtractor.QuantizationParams(0.006008731201291084f, 123),
        ),
    )

    private var mediaProjectionManager: MediaProjectionManager? = null
    private var mediaProjection: MediaProjection? = null
    private var mMediaProjectionCallback: MediaProjectionCallback? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private var mScreenDensity: Int = 0
    private var mDisplayWidth: Int = 1280
    private var mDisplayHeight: Int = 800

    private var fps : Int = 0
    private var startDateTime: LocalDateTime = LocalDateTime.now()

    private var inferenceTimeSum : Long = 0
    private var inferenceIndex : Int = 0
    private val inferenceSamples : Int = 200
    private var calculatedMeanInf: Long = 0

    inner class MediaProjectionCallback : MediaProjection.Callback() {
        override fun onStop() {
            mediaProjection = null
            //stopScreenSharing()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        selectedModel = models.first()

        tfliteDetector = TfliteDetector()
        tfliteDetector.initializeModel(this, selectedModel)

        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        mediaProjectionManager =
            applicationContext.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager?

        linkElements()
        setVideoView()
        setSpinner()

        setScreenSpecs()

        boxPaint.strokeWidth = 5f
        boxPaint.style = Paint.Style.STROKE
        boxPaint.color = Color.RED

        textPain.textSize = 50f
        textPain.color = Color.GREEN
        textPain.style = Paint.Style.FILL

        requestNotificationsPermissions()
    }

    private fun linkElements(){
        videoView = findViewById(R.id.videoView)
        imageView = findViewById(R.id.imageView)
        spinner = findViewById(R.id.spinner)
        fpsText = findViewById(R.id.fps)
        inferenceText = findViewById(R.id.inference)
        totalTimeText = findViewById(R.id.totalTime)
        infMeanText = findViewById(R.id.infMean)
    }

    private fun setVideoView(){
        val videoPath = "android.resource://" + packageName + "/" + selectedModel.videoExample
        val uri = Uri.parse(videoPath)
        videoView.setVideoURI(uri)

        videoView.start()

        videoView.setOnCompletionListener(MediaPlayer.OnCompletionListener {
            videoView.start() // Restart the video to create a loop
        })
    }

    private fun setSpinner() {
        val adapter = SpinnerAdapter(this, models)

        spinner.adapter = adapter

        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                selectedModel = parent.getItemAtPosition(position) as AIModel
                setVideoView()
                tfliteDetector.initializeModel(applicationContext, selectedModel)
                resetFps()
                resetInferenceTimes()
            }

            override fun onNothingSelected(parent: AdapterView<*>) {
            }
        }
    }

    private fun requestNotificationsPermissions(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED){
            requestScreenCapture()
        }else{
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                NOTIFICATION_PERMISSION_CODE
            )
        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if(requestCode == NOTIFICATION_PERMISSION_CODE){
            if(grantResults.all { it == 0 }){
                requestScreenCapture()
            }

        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun requestScreenCapture(){
        Intent(applicationContext,ImageStreamService::class.java).also {
            it.action = ImageStreamService.Actions.START.toString()
            startService(it)
        }
        val permissionIntent = mediaProjectionManager?.createScreenCaptureIntent()
        ActivityCompat.startActivityForResult(
            this,
            permissionIntent!!,
            SCREEN_CAPTURE_PERMISSION_CODE,
            null
        )
    }

    fun dpToPx(dp: Int): Int {
        return (dp / Resources.getSystem().displayMetrics.ydpi).toInt()
    }

    // Function to crop the bitmap
    fun cropBitmapToCenter(originalBitmap: Bitmap, targetHeightDp: Int): Bitmap {
        val targetHeightPx = dpToPx(targetHeightDp)

        // Calculate the width of the ImageView proportionally
        val aspectRatio = originalBitmap.width.toFloat() / originalBitmap.height.toFloat()
        val targetWidthPx = (targetHeightPx * aspectRatio).toInt()

        // Calculate the cropping bounds
        val startX = (originalBitmap.width - targetWidthPx) / 2
        val startY = (originalBitmap.height - targetHeightPx) / 2

        return Bitmap.createBitmap(originalBitmap, startX, startY, targetWidthPx, targetHeightPx)
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == SCREEN_CAPTURE_PERMISSION_CODE){
            imageReader = ImageReader.newInstance(mDisplayWidth, mDisplayHeight, PixelFormat.RGBA_8888, 2)
            mMediaProjectionCallback = MediaProjectionCallback()
            mediaProjection = mediaProjectionManager?.getMediaProjection(resultCode, data!!)
            mediaProjection?.registerCallback(mMediaProjectionCallback!!, null)
            virtualDisplay = createVirtualDisplay()
            resetFps()
            resetInferenceTimes()
            imageReader!!.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                if (image != null) {
                    val startTotalTimeInference = LocalDateTime.now()

                    val planes = image.planes
                    val buffer = planes[0].buffer
                    val pixelStride = planes[0].pixelStride
                    val rowStride = planes[0].rowStride
                    val rowPadding = rowStride - pixelStride * mDisplayWidth

                    val bitmap = Bitmap.createBitmap( mDisplayWidth, mDisplayHeight, Bitmap.Config.ARGB_8888)
                    bitmap.copyPixelsFromBuffer(buffer)

                    val bitmapWidth = bitmap.width
                    val bitmapHeight = bitmap.height

                    val croppedBitmap =  Bitmap.createBitmap(
                        bitmap,
                        0,
                        (bitmapHeight/2) - (imageView.height/2),
                        bitmapWidth,
                        imageView.height
                    );


                    val detectionResult: DetectionResult = tfliteDetector.detect(croppedBitmap)

                    inferenceText.text = "inf: ${detectionResult.inferenceTime.toMillis()} ms"


                    val mutableBitmap = croppedBitmap.copy(Bitmap.Config.ARGB_8888, true)
                    val canvas = Canvas(mutableBitmap)

                    for (recognition in detectionResult.recognitions) {
                        if (recognition.confidence > 0.4) {
                            val location: RectF = recognition.location
                            canvas.drawRect(location, boxPaint)
                            canvas.drawText(
                                recognition.name + ":" + String.format ("%.1f",(recognition.confidence)*100) + "%",
                                location.left,
                                location.top,
                                textPain
                            )
                        }
                    }

                    imageView.setImageBitmap(mutableBitmap)

                    image.close()
                    calculateInference(startTotalTimeInference, detectionResult.inferenceTime.toMillis())
                    calculateFps()
                }
            }, null)
        }
    }

    private fun createVirtualDisplay(): VirtualDisplay? {
        try {
            return mediaProjection?.createVirtualDisplay(
                "MainActivity", mDisplayWidth, mDisplayHeight, mScreenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, imageReader?.surface, null, null
            )
        } catch (e: Exception) {
            println("createVirtualDisplay err")
            println(e.message)
            return null
        }
    }

    private fun setScreenSpecs(){
        val metrics = DisplayMetrics()

        val display = this.display

        display?.getRealMetrics(metrics)

        mScreenDensity = metrics.densityDpi

        mDisplayHeight = metrics.heightPixels
        mDisplayWidth = metrics.widthPixels

        var maxRes = 1280.0;
        if (metrics.scaledDensity >= 3.0f) {
            maxRes = 1920.0;
        }
        if (metrics.widthPixels > metrics.heightPixels) {
            var rate = metrics.widthPixels / maxRes

            if (rate > 1.5) {
                rate = 1.5
            }
            mDisplayWidth = maxRes.toInt()
            mDisplayHeight = (metrics.heightPixels / rate).toInt()
            println("Rate : $rate")
        } else {
            var rate = metrics.heightPixels / maxRes
            if (rate > 1.5) {
                rate = 1.5
            }
            mDisplayHeight = maxRes.toInt()
            mDisplayWidth = (metrics.widthPixels / rate).toInt()
            println("Rate : $rate")
        }

        println("Scaled Density")
        println(metrics.scaledDensity)
        println("Original Resolution ")
        println(metrics.widthPixels.toString() + " x " + metrics.heightPixels)
        println("Calcule Resolution ")
        println("$mDisplayWidth x $mDisplayHeight")
    }

    private fun resetFps(){
        fps = 0
        startDateTime = LocalDateTime.now()
    }

    private fun calculateFps(){
        val now = LocalDateTime.now()
        val timeDiff = Duration.between(startDateTime, now).toMillis()
        if(timeDiff > MILLIS_TO_CALCULATE_FPS){
            val evalFps = String.format ("%.1f",(fps.toDouble()/timeDiff.toDouble())*1000)
            fpsText.text = "fps: ${evalFps}"
            resetFps()
        }else{
            fps++
        }
    }

    private fun resetInferenceTimes(){
        inferenceIndex = 0
        inferenceTimeSum = 0L
        calculatedMeanInf = 0L
    }
    private fun calculateInference(startTotalTimeInference: LocalDateTime, inferenceTime: Long) {
        val timeDiff = Duration.between(startTotalTimeInference, LocalDateTime.now()).toMillis()
        totalTimeText.text = "total: ${timeDiff} ms"
        inferenceTimeSum += inferenceTime
        inferenceIndex ++
        if(inferenceIndex >= inferenceSamples){
            calculatedMeanInf = inferenceTimeSum/inferenceIndex
            inferenceIndex = 0
            inferenceTimeSum = 0L
        }
        infMeanText.text = "${inferenceIndex}/${inferenceSamples}: ${selectedModel.name} ${calculatedMeanInf}"
    }
}