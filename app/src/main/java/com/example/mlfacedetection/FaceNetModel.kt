package com.example.mlfacedetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
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

// VERSI DENGAN "IDE 3 LITE" (Grayscale Preprocessing)
class FaceNetModel(context: Context) {

    private val interpreter: Interpreter
    private val imageProcessor: ImageProcessor

    private val imageSize = 112
    private val embeddingDim = 192

    init {
        val modelBuffer = FileUtil.loadMappedFile(context, "mobile_face_net.tflite")
        val options = Interpreter.Options()
        interpreter = Interpreter(modelBuffer, options)

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f)) // Normalisasi ke [-1.0, 1.0]
            .build()
    }

    fun getFaceEmbedding(image: Bitmap, crop: Rect): FloatArray {
        // 1. Crop wajah dari gambar utuh (Kode Anda sudah ada)
        val croppedBitmap = cropRectFromBitmap(image, crop)

        // =========================================================
        // ▼▼▼ IDE 3 "LITE": KONVERSI KE GRAYSCALE ▼▼▼
        // =========================================================
        // Kita ubah bitmap hasil crop menjadi hitam putih
        // Ini membantu mengurangi variasi karena pencahayaan berwarna
        val grayscaleBitmap = toGrayscale(croppedBitmap)
        // =========================================================

        // 2. Gunakan TensorImage dan ImageProcessor
        val tensorImage = TensorImage(DataType.FLOAT32)

        // 3. Muat bitmap yang SUDAH di-grayscale
        tensorImage.load(grayscaleBitmap)

        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        // 4. Siapkan buffer untuk output
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, embeddingDim), DataType.FLOAT32)

        // 5. Jalankan model
        try {
            interpreter.run(imageBuffer, outputBuffer.buffer)
        } catch (e: Exception) {
            Log.e("FaceNetModel", "Error running model inference: ${e.message}")
        }

        // 6. Kembalikan hasilnya
        return outputBuffer.floatArray
    }

    // --- FUNGSI BARU UNTUK KONVERSI GRAYSCALE ---
    private fun toGrayscale(bmpOriginal: Bitmap): Bitmap {
        val width = bmpOriginal.width
        val height = bmpOriginal.height
        val bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmpGrayscale)
        val paint = Paint()
        val cm = ColorMatrix()
        cm.setSaturation(0f) // 0 = Grayscale, 1 = Normal
        val f = ColorMatrixColorFilter(cm)
        paint.colorFilter = f
        c.drawBitmap(bmpOriginal, 0f, 0f, paint)
        return bmpGrayscale
    }
    // ---------------------------------------------

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