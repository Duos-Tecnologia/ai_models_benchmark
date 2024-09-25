package com.example.androidscreenstreamer


import android.app.Notification
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.example.ai_models_benchmark.R

class ImageStreamService : Service() {

    private val notificationId = 1
    private val notificationChannelId = "running_channel"

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            Actions.START.toString() -> start()
            Actions.STOP.toString() -> stopSelf()
        }
        return super.onStartCommand(intent, flags, startId)
    }

    private fun start() {
        // Create initial notification
        val notification = createNotification("Running")

        startForeground(notificationId, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)

    }

    private fun createNotification(contentText: String): Notification {
        return NotificationCompat.Builder(this, notificationChannelId)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Running")
            .setContentText(contentText)
            .setOnlyAlertOnce(true)  // Ensure the notification doesn't make noise/vibrate when updated
            .setOngoing(true).build()
    }

    private fun updateNotification(seconds: Int) {
        val minutes = seconds / 60
        val secondsDisplay = seconds % 60
        val contentText = String.format("Elapsed time is %02d:%02d", minutes, secondsDisplay)

        // Create the updated notification
        val updatedNotification = createNotification(contentText)

        // Update the notification
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as android.app.NotificationManager
        notificationManager.notify(notificationId, updatedNotification)

    }

    enum class Actions {
        START, STOP
    }
}
