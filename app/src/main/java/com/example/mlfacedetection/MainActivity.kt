@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalGetImage::class)

package com.example.mlfacedetection // Pastikan ini adalah package Anda

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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset // Pastikan import ini ada
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
import io.ktor.client.plugins.*
import io.ktor.http.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.OptIn

// --- State untuk Kontrol ---
enum class AppState { IDLE, LOADING }

// --- State untuk Liveness (Ide 1) ---
enum class LivenessState { IDLE, WAITING_FOR_BLINK, BLINK_DETECTED }

// --- State untuk Enrollment (VERSI 4-ANGLE BARU) ---
enum class EnrollmentState {
    IDLE, // Menunggu tombol 'mulai'

    // 4 Langkah baru
    GET_CENTER_NEUTRAL, SENDING_CENTER_NEUTRAL,
    GET_CENTER_SMILE, SENDING_CENTER_SMILE,
    GET_LEFT, SENDING_LEFT,
    GET_RIGHT, SENDING_RIGHT,

    FINISHED // Selesai
}


class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) { setContent { MLFaceDetectionTheme { MainAppScreen(cameraExecutor) } }
        } else { setContent { MLFaceDetectionTheme { PermissionDeniedScreen() } } }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) -> {
                setContent { MLFaceDetectionTheme { MainAppScreen(cameraExecutor) } }
            }
            else -> requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

// --- Composable Utama yang Mengatur Tampilan ---
@SuppressLint("UnrememberedMutableState")
@Composable
fun MainAppScreen(cameraExecutor: ExecutorService) {
    var presensiResult by remember { mutableStateOf<RecognizeResponse?>(null) }

    if (presensiResult != null) {
        AttendanceResultScreen(
            result = presensiResult!!,
            onBack = { presensiResult = null }
        )
    } else {
        FaceDetectionScreen(
            cameraExecutor = cameraExecutor,
            onRecognitionSuccess = { response ->
                presensiResult = response
            }
        )
    }
}

@androidx.annotation.OptIn(ExperimentalGetImage::class)
@SuppressLint("UnrememberedMutableState")
@Composable
fun FaceDetectionScreen(
    cameraExecutor: ExecutorService,
    onRecognitionSuccess: (RecognizeResponse) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var detectedFaces by remember { mutableStateOf<List<Face>>(emptyList()) }
    var faceEmbedding by remember { mutableStateOf<FloatArray?>(null) }
    var imageSize by remember { mutableStateOf(Size(0f, 0f)) }
    val faceNetModel = remember { FaceNetModel(context) }
    val coroutineScope = rememberCoroutineScope()

    // --- State UI ---
    var selectedTabIndex by remember { mutableStateOf(0) }
    var appState by remember { mutableStateOf(AppState.IDLE) }
    var currentMessage by remember { mutableStateOf("") }

    // --- State untuk Form Daftar ---
    var regUserId by remember { mutableStateOf("") }
    var regUserName by remember { mutableStateOf("") }
    var enrollmentState by remember { mutableStateOf(EnrollmentState.IDLE) }
    var enrollmentMessage by remember { mutableStateOf("") }

    // --- State untuk Liveness & Angle (Ide 1 & 2) ---
    var headEulerAngleX by remember { mutableStateOf(0f) }
    var headEulerAngleY by remember { mutableStateOf(0f) }
    var leftEyeOpenProb by remember { mutableStateOf(1f) }
    var rightEyeOpenProb by remember { mutableStateOf(1f) }
    var smileProb by remember { mutableStateOf(0f) } // <-- STATE BARU

    var livenessState by remember { mutableStateOf(LivenessState.IDLE) }


    val faceDetector = remember {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL) // Untuk Liveness (Mata) & Senyum
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL) // Untuk Angle
            .build()
        FaceDetection.getClient(options)
    }

    val registerAngle: (FloatArray, EnrollmentState) -> Unit = { embedding, nextState ->
        enrollmentMessage = "Bagus! Angle terambil. Mengirim..."
        coroutineScope.launch {
            try {
                ApiService.registerFace(regUserId, regUserName, embedding.toList())
                enrollmentState = nextState
                enrollmentMessage = ""
            } catch (e: Exception) {
                enrollmentMessage = "Gagal kirim. Coba lagi."
                enrollmentState = EnrollmentState.values()[enrollmentState.ordinal - 1] // Kembali
            }
        }
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
                                    val firstFace = faces.first()

                                    // --- BACA SEMUA DATA ---
                                    headEulerAngleX = firstFace.headEulerAngleX
                                    headEulerAngleY = firstFace.headEulerAngleY
                                    firstFace.leftEyeOpenProbability?.let { leftEyeOpenProb = it }
                                    firstFace.rightEyeOpenProbability?.let { rightEyeOpenProb = it }
                                    firstFace.smilingProbability?.let { smileProb = it } // <-- BACA SENYUM

                                    val originalBitmap = imageProxy.toBitmap()
                                    if (originalBitmap != null) {
                                        val softwareBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
                                        faceEmbedding = faceNetModel.getFaceEmbedding(softwareBitmap, firstFace.boundingBox)
                                    }

                                    // --- Cek Liveness (jika diminta) ---
                                    if (livenessState == LivenessState.WAITING_FOR_BLINK) {
                                        if (leftEyeOpenProb < 0.3 && rightEyeOpenProb < 0.3) {
                                            livenessState = LivenessState.BLINK_DETECTED
                                        }
                                    }

                                    // --- Cek Wizard Pendaftaran (jika sedang berjalan) ---
                                    val currentEmbedding = faceEmbedding
                                    if (currentEmbedding != null && appState == AppState.IDLE) {

                                        // Variabel helper
                                        val isSmiling = smileProb > 0.8 // 80% smile
                                        val isLookingLeft = headEulerAngleY > 30
                                        val isLookingRight = headEulerAngleY < -30
                                        val isLookingCenterH = headEulerAngleY > -10 && headEulerAngleY < 10

                                        when (enrollmentState) {
                                            EnrollmentState.GET_CENTER_NEUTRAL -> {
                                                if (isLookingCenterH && !isSmiling) {
                                                    enrollmentState = EnrollmentState.SENDING_CENTER_NEUTRAL
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_CENTER_SMILE)
                                                }
                                            }
                                            EnrollmentState.GET_CENTER_SMILE -> {
                                                if (isLookingCenterH && isSmiling) {
                                                    enrollmentState = EnrollmentState.SENDING_CENTER_SMILE
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_LEFT)
                                                }
                                            }
                                            EnrollmentState.GET_LEFT -> {
                                                if (isLookingLeft) {
                                                    enrollmentState = EnrollmentState.SENDING_LEFT
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_RIGHT)
                                                }
                                            }
                                            EnrollmentState.GET_RIGHT -> {
                                                if (isLookingRight) {
                                                    enrollmentState = EnrollmentState.SENDING_RIGHT
                                                    registerAngle(currentEmbedding, EnrollmentState.FINISHED)
                                                }
                                            }
                                            else -> {}
                                        }
                                    }

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

    Scaffold(
        topBar = { TopAppBar(title = { Text("Presensi Wajah") }, colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary, titleContentColor = MaterialTheme.colorScheme.onPrimary)) }
    ) { paddingValues ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
        ) {

            // =========================================================
            // ▼▼▼ BOX KAMERA SEKARANG DENGAN OVAL GUIDE ▼▼▼
            // =========================================================
            Box(modifier = Modifier
                .fillMaxWidth()
                .height(400.dp)) {

                // 1. Kamera (di paling bawah)
                CameraView(modifier = Modifier.fillMaxSize(), analyzer = imageAnalyzer, lifecycleOwner = lifecycleOwner)

                // 2. Overlay Wajah (Kuning)
                FaceOverlay(modifier = Modifier.fillMaxSize(), faces = detectedFaces, imageSize = imageSize)

                // 3. OVAL GUIDE (di atas segalanya)
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val ovalWidth = size.width * 0.7f // 70% lebar
                    val ovalHeight = ovalWidth * 1.3f // oval potrait

                    drawOval(
                        color = Color.White.copy(alpha = 0.6f),
                        topLeft = Offset(
                            x = (size.width - ovalWidth) / 2, // Center H
                            y = (size.height - ovalHeight) / 2 // Center V
                        ),
                        size = androidx.compose.ui.geometry.Size(ovalWidth, ovalHeight),
                        style = Stroke(width = 4.dp.toPx()) // Garis
                    )
                }

                // 4. Teks Status
                Text(
                    text = if (faceEmbedding != null) "Wajah Terdeteksi" else "Arahkan ke Wajah",
                    color = Color.White, modifier = Modifier.align(Alignment.TopCenter).padding(8.dp)
                )
            }
            // =========================================================
            // ▲▲▲ AKHIR DARI BOX KAMERA ▲▲▲
            // =========================================================

            TabRow(selectedTabIndex = selectedTabIndex) {
                Tab(
                    selected = selectedTabIndex == 0,
                    onClick = { selectedTabIndex = 0; currentMessage = "" },
                    text = { Text("PRESENSI") }
                )
                Tab(
                    selected = selectedTabIndex == 1,
                    onClick = {
                        selectedTabIndex = 1
                        currentMessage = ""
                        enrollmentState = EnrollmentState.IDLE
                        enrollmentMessage = ""
                    },
                    text = { Text("DAFTAR WAJAH") }
                )
            }

            if (appState == AppState.LOADING || enrollmentState.name.startsWith("SENDING")) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            Column(modifier = Modifier.padding(16.dp)) {
                if (currentMessage.isNotBlank()) {
                    Text(currentMessage, fontWeight = FontWeight.Bold, color = if (currentMessage.startsWith("Error")) Color.Red else Color(0xFF4CAF50))
                    Spacer(modifier = Modifier.height(16.dp))
                }

                when (selectedTabIndex) {
                    // === TAB PRESENSI ===
                    0 -> Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {

                        if (livenessState == LivenessState.WAITING_FOR_BLINK) {
                            Text("Posisikan wajah di OVAL, lalu KEDIPKAN MATA ANDA",
                                color = Color.Blue, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
                        } else {
                            Text("Posisikan wajah Anda di dalam OVAL dan tekan tombol di bawah.")
                        }

                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = {
                                val embeddingAtClick = faceEmbedding
                                if (embeddingAtClick == null) {
                                    currentMessage = "Error: Wajah hilang. Posisikan di oval."
                                    return@Button
                                }

                                currentMessage = "Memulai Liveness Check..."
                                livenessState = LivenessState.WAITING_FOR_BLINK
                                appState = AppState.LOADING

                                coroutineScope.launch {
                                    var timeout = 0
                                    while (livenessState != LivenessState.BLINK_DETECTED && timeout < 100) {
                                        delay(50)
                                        timeout++
                                    }

                                    if (livenessState == LivenessState.BLINK_DETECTED) {
                                        currentMessage = "Liveness Sukses! Mencocokkan..."
                                        try {
                                            val response = ApiService.recognizeFace(embeddingAtClick.toList())
                                            onRecognitionSuccess(response)
                                        } catch (e: ClientRequestException) {
                                            currentMessage = if (e.response.status == HttpStatusCode.NotFound) {
                                                "Error: Wajah tidak dikenali. (Skor di bawah 90%)"
                                            } else {
                                                "Error Server: ${e.response.status.description}"
                                            }
                                            appState = AppState.IDLE
                                        } catch (e: Exception) {
                                            currentMessage = "Error Koneksi: Periksa jaringan Anda."
                                            appState = AppState.IDLE
                                        }
                                    } else {
                                        currentMessage = "Error: Liveness Gagal. Anda tidak berkedip."
                                        appState = AppState.IDLE
                                    }

                                    livenessState = LivenessState.IDLE
                                }
                            },
                            enabled = faceEmbedding != null && appState == AppState.IDLE,
                            modifier = Modifier.fillMaxWidth().height(48.dp)
                        ) {
                            Text("LAKUKAN PRESENSI SEKARANG")
                        }
                    }

                    // === TAB DAFTAR (Ide 2 - 4 Angle) ===
                    1 -> Column(modifier = Modifier.fillMaxWidth()) {

                        OutlinedTextField(
                            value = regUserId,
                            onValueChange = { regUserId = it },
                            label = { Text("User ID (NPM/Email)") },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = enrollmentState == EnrollmentState.IDLE
                        )
                        Spacer(modifier = Modifier.height(8.dp))

                        OutlinedTextField(
                            value = regUserName,
                            onValueChange = { regUserName = it },
                            label = { Text("Nama Lengkap") },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = enrollmentState == EnrollmentState.IDLE
                        )
                        Spacer(modifier = Modifier.height(16.dp))

                        Button(
                            onClick = {
                                if(regUserId.isBlank() || regUserName.isBlank()) {
                                    currentMessage = "Error: Isi ID dan Nama lebih dulu."
                                } else {
                                    enrollmentState = EnrollmentState.GET_CENTER_NEUTRAL
                                    currentMessage = ""
                                }
                            },
                            enabled = appState == AppState.IDLE && enrollmentState == EnrollmentState.IDLE && faceEmbedding != null
                        ) {
                            Text("Mulai Pendaftaran Terpandu (4 Angle)")
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        val instructionColor = Color.Blue
                        if (enrollmentMessage.isNotBlank()) {
                            Text(enrollmentMessage, modifier = Modifier.padding(top = 8.dp))
                        }

                        when (enrollmentState) {
                            EnrollmentState.IDLE -> {
                                Text("Posisikan wajah Anda di dalam OVAL, lalu tekan 'Mulai'.")
                            }
                            EnrollmentState.GET_CENTER_NEUTRAL, EnrollmentState.SENDING_CENTER_NEUTRAL -> {
                                Text("1/4: Wajah di OVAL, lihat LURUS & NETRAL...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_CENTER_SMILE, EnrollmentState.SENDING_CENTER_SMILE -> {
                                Text("2/4: Wajah di OVAL, lihat LURUS & SENYUM LEBAR!", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_LEFT, EnrollmentState.SENDING_LEFT -> {
                                Text("3/4: Wajah di OVAL, TENGOK KE KIRI...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_RIGHT, EnrollmentState.SENDING_RIGHT -> {
                                Text("4/4: Wajah di OVAL, TENGOK KE KANAN...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.FINISHED -> {
                                Text("Pendaftaran Selesai! 4 angle telah tersimpan.", fontWeight = FontWeight.Bold, color = Color(0xFF4CAF50))
                                Button(onClick = { enrollmentState = EnrollmentState.IDLE }) {
                                    Text("Daftarkan Ulang / Tambah Angle")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// --- Composable BARU untuk Layar Sukses ---
@Composable
fun AttendanceResultScreen(result: RecognizeResponse, onBack: () -> Unit) {
    Box(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("PRESENSI SUKSES!", style = MaterialTheme.typography.headlineMedium, color = Color(0xFF4CAF50), fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(16.dp))
            Text("Selamat datang,", style = MaterialTheme.typography.titleLarge)
            Text(result.name, style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            Text("ID: ${result.user_id}", style = MaterialTheme.typography.bodyLarge)
            Text("Kecocokan: ${"%.2f".format(result.similarity * 100)}%", style = MaterialTheme.typography.bodyLarge)
            Spacer(modifier = Modifier.height(32.dp))
            Button(onClick = onBack, modifier = Modifier.fillMaxWidth()) {
                Text("OK")
            }
        }
    }
}


// --- Composable Bawaan (Tidak Berubah) ---

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
fun PermissionDeniedScreen() {
    Box(modifier = Modifier.fillMaxSize().padding(16.dp), contentAlignment = Alignment.Center) {
        Text("Izin kamera ditolak. Aplikasi ini memerlukan izin kamera untuk mendeteksi wajah. Silakan aktifkan izin melalui pengaturan aplikasi.", textAlign = TextAlign.Center, style = MaterialTheme.typography.bodyLarge)
    }
}