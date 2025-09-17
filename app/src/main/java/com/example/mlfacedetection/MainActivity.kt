@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalGetImage::class)

package com.example.mlfacedetection

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.mlfacedetection.ui.theme.MLFaceDetectionTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.OptIn

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) { setContent { MLFaceDetectionTheme { FaceDetectionScreen(cameraExecutor) } }
        } else { setContent { MLFaceDetectionTheme { PermissionDeniedScreen() } } }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) -> {
                setContent { MLFaceDetectionTheme { FaceDetectionScreen(cameraExecutor) } }
            }
            else -> requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

@androidx.annotation.OptIn(ExperimentalGetImage::class)
@SuppressLint("UnrememberedMutableState")
@Composable
fun FaceDetectionScreen(cameraExecutor: ExecutorService) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var detectedFaces by remember { mutableStateOf<List<Face>>(emptyList()) }
    var faceEmbedding by remember { mutableStateOf<FloatArray?>(null) }
    var imageSize by remember { mutableStateOf(Size(0f, 0f)) }
    val faceNetModel = remember { FaceNetModel(context) }
    val faceDetector = remember {
        val options = FaceDetectorOptions.Builder().setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST).build()
        FaceDetection.getClient(options)
    }

    val imageAnalyzer = remember {
        ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    val mediaImage = imageProxy.image
                    if (mediaImage != null) {
                        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                        imageSize = Size(imageProxy.width.toFloat(), imageProxy.height.toFloat())

                        faceDetector.process(image)
                            .addOnSuccessListener { faces ->
                                detectedFaces = faces
                                if (faces.isNotEmpty()) {
                                    // =========================================================
                                    // PERUBAHAN UTAMA DI SINI
                                    // =========================================================
                                    val originalBitmap = imageProxy.toBitmap() // 1. Panggil fungsi bawaan
                                    if (originalBitmap != null) {
                                        // 2. Salin ke format yang aman (Software)
                                        val softwareBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
                                        val firstFace = faces.first()
                                        // 3. Kirim bitmap yang aman ke model
                                        faceEmbedding = faceNetModel.getFaceEmbedding(softwareBitmap, firstFace.boundingBox)
                                    }
                                    // =========================================================
                                } else {
                                    faceEmbedding = null
                                }
                            }
                            .addOnFailureListener { e -> Log.e("FaceDetection", "Deteksi wajah gagal", e) }
                            .addOnCompleteListener { imageProxy.close() }
                    } else {
                        imageProxy.close()
                    }
                }
            }
    }

    Scaffold(topBar = { TopAppBar(title = { Text("Face Embedding Extractor") }, colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary, titleContentColor = MaterialTheme.colorScheme.onPrimary)) }) { paddingValues ->
        Box(modifier = Modifier.fillMaxSize().padding(paddingValues)) {
            CameraView(modifier = Modifier.fillMaxSize(), analyzer = imageAnalyzer, lifecycleOwner = lifecycleOwner)
            FaceOverlay(modifier = Modifier.fillMaxSize(), faces = detectedFaces, imageSize = imageSize)
            FaceDataCard(modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp), faces = detectedFaces, embedding = faceEmbedding)
        }
    }
}

// ... Composable dan fungsi lainnya tidak berubah ...
@Composable
fun CameraView(modifier: Modifier = Modifier, analyzer: ImageAnalysis, lifecycleOwner: androidx.lifecycle.LifecycleOwner) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember { PreviewView(context) }
    AndroidView(factory = { previewView }, modifier = modifier) {
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, analyzer)
            } catch (exc: Exception) {
                Log.e("CameraView", "Gagal bind use cases", exc)
            }
        }, ContextCompat.getMainExecutor(context))
    }
}
@Composable
fun FaceOverlay(modifier: Modifier = Modifier, faces: List<Face>, imageSize: Size) {
    Canvas(modifier = modifier) {
        if (imageSize.width == 0f || imageSize.height == 0f) return@Canvas
        val scaleX = size.width / imageSize.height
        val scaleY = size.height / imageSize.width
        faces.forEach { face ->
            val boundingBox = face.boundingBox
            drawRect(
                color = Color.Yellow,
                topLeft = androidx.compose.ui.geometry.Offset(x = size.width - (boundingBox.exactCenterX() * scaleX) - (boundingBox.width() / 2f * scaleX), y = boundingBox.exactCenterY() * scaleY - (boundingBox.height() / 2f * scaleY)),
                size = androidx.compose.ui.geometry.Size(width = boundingBox.width() * scaleX, height = boundingBox.height() * scaleY),
                style = Stroke(width = 2.dp.toPx())
            )
        }
    }
}
@Composable
fun FaceDataCard(modifier: Modifier = Modifier, faces: List<Face>, embedding: FloatArray?) {
    if (faces.isNotEmpty()) {
        Card(
            modifier = modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = 0.7f)),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(text = "Wajah Terdeteksi!", color = Color(0xFF4CAF50), fontWeight = FontWeight.Bold, fontSize = 18.sp)
                Spacer(modifier = Modifier.height(8.dp))
                if (embedding != null) {
                    Text(text = "Face Embedding (192-D Vector):", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    val embeddingPreview = embedding.take(5).joinToString(", ") { "%.3f".format(it) }
                    Text(text = "[$embeddingPreview, ...]", color = Color.White, fontSize = 12.sp)
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp, color = Color.Yellow)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Mengekstrak embedding...", color = Color.Yellow, fontSize = 14.sp)
                    }
                }
            }
        }
    }
}
@Composable
fun PermissionDeniedScreen() {
    Box(modifier = Modifier.fillMaxSize().padding(16.dp), contentAlignment = Alignment.Center) {
        Text("Izin kamera ditolak. Aplikasi ini memerlukan izin kamera untuk mendeteksi wajah. Silakan aktifkan izin melalui pengaturan aplikasi.", textAlign = TextAlign.Center, style = MaterialTheme.typography.bodyLarge)
    }
}

// KITA SUDAH TIDAK PERLU FUNGSI toBitmapManually() LAGI
// HAPUS FUNGSI ITU DARI KODEMU