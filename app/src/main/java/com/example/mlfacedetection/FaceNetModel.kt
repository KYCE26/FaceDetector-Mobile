package com.example.mlfacedetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

// VERSI FINAL - Menggunakan TFLite Support Library
class FaceNetModel(context: Context) {

    private val interpreter: Interpreter
    private val imageProcessor: ImageProcessor

    private val imageSize = 112
    private val embeddingDim = 192

    init {
        val modelBuffer = FileUtil.loadMappedFile(context, "mobile_face_net.tflite")
        val options = Interpreter.Options()
        // Kita bisa tambahkan delegasi GPU untuk percepatan jika perangkat mendukung
        // options.addDelegate(GpuDelegate())
        interpreter = Interpreter(modelBuffer, options)

        // ImageProcessor untuk melakukan resize dan normalisasi secara otomatis
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f)) // Normalisasi ke [-1.0, 1.0]
            .build()
    }

    fun getFaceEmbedding(image: Bitmap, crop: Rect): FloatArray {
        // 1. Crop wajah dari gambar utuh
        val croppedBitmap = cropRectFromBitmap(image, crop)

        // 2. Gunakan TensorImage dan ImageProcessor (lebih modern dan aman)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(croppedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        // 3. Siapkan buffer untuk output
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, embeddingDim), DataType.FLOAT32)

        // 4. Jalankan model
        try {
            interpreter.run(imageBuffer, outputBuffer.buffer)
        } catch (e: Exception) {
            Log.e("FaceNetModel", "Error running model inference: ${e.message}")
        }

        // 5. Kembalikan hasilnya
        return outputBuffer.floatArray
    }

    private fun cropRectFromBitmap(source: Bitmap, rect: Rect): Bitmap {
        var width = rect.width()
        var height = rect.height()
        if (rect.left < 0) rect.left = 0
        if (rect.top < 0) rect.top = 0
        if (rect.right > source.width) rect.right = source.width
        if (rect.bottom > source.height) rect.bottom = source.height
        width = rect.width()
        height = rect.height()

        if (width <= 0 || height <= 0) return source

        return Bitmap.createBitmap(
            source,
            rect.left,
            rect.top,
            width,
            height
        )
    }
}