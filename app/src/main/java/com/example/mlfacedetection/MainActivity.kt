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
enum class AppState { IDLE, LOADING } // Status global aplikasi

// --- State untuk Liveness (Ide 1) ---
enum class LivenessState { IDLE, WAITING_FOR_BLINK, BLINK_DETECTED }

// --- State untuk Enrollment (Ide 2 - VERSI 9 ANGLE) ---
enum class EnrollmentState {
    IDLE, // Menunggu tombol 'mulai'

    // Baris Tengah
    GET_FRONT_CENTER, SENDING_FRONT_CENTER,
    GET_FRONT_LEFT, SENDING_FRONT_LEFT,
    GET_FRONT_RIGHT, SENDING_FRONT_RIGHT,

    // Baris Atas (Melihat ke Atas)
    GET_TOP_CENTER, SENDING_TOP_CENTER,
    GET_TOP_LEFT, SENDING_TOP_LEFT,
    GET_TOP_RIGHT, SENDING_TOP_RIGHT,

    // Baris Bawah (Melihat ke Bawah)
    GET_BOTTOM_CENTER, SENDING_BOTTOM_CENTER,
    GET_BOTTOM_LEFT, SENDING_BOTTOM_LEFT,
    GET_BOTTOM_RIGHT, SENDING_BOTTOM_RIGHT,

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
    val context = LocalContext.current // <-- Perbaikan dari error sebelumnya
    val lifecycleOwner = LocalLifecycleOwner.current
    var detectedFaces by remember { mutableStateOf<List<Face>>(emptyList()) }
    var faceEmbedding by remember { mutableStateOf<FloatArray?>(null) }
    var imageSize by remember { mutableStateOf(Size(0f, 0f)) }
    val faceNetModel = remember { FaceNetModel(context) } // <-- Perbaikan dari error sebelumnya
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
    var headEulerAngleX by remember { mutableStateOf(0f) } // Atas/Bawah
    var headEulerAngleY by remember { mutableStateOf(0f) } // Kiri/Kanan
    var leftEyeOpenProb by remember { mutableStateOf(1f) }
    var rightEyeOpenProb by remember { mutableStateOf(1f) }
    var livenessState by remember { mutableStateOf(LivenessState.IDLE) }


    val faceDetector = remember {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL) // Untuk Liveness (Mata)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL) // Untuk Angle (X dan Y)
            .build()
        FaceDetection.getClient(options)
    }

    // --- Helper Lambda untuk Mengirim Data Pendaftaran ---
    // Ini untuk menghindari duplikasi kode
    val registerAngle: (FloatArray, EnrollmentState) -> Unit = { embedding, nextState ->
        enrollmentMessage = "Bagus! Angle terambil. Mengirim..."
        coroutineScope.launch {
            try {
                ApiService.registerFace(regUserId, regUserName, embedding.toList())
                enrollmentState = nextState // Lanjut ke state berikutnya
                enrollmentMessage = "" // Hapus pesan 'mengirim'
            } catch (e: Exception) {
                enrollmentMessage = "Gagal kirim. Coba lagi."
                enrollmentState = EnrollmentState.values()[enrollmentState.ordinal - 1]
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

                                    // --- BACA DATA BARU (Angle X dan Y) ---
                                    headEulerAngleX = firstFace.headEulerAngleX // Atas/Bawah
                                    headEulerAngleY = firstFace.headEulerAngleY // Kiri/Kanan
                                    firstFace.leftEyeOpenProbability?.let { leftEyeOpenProb = it }
                                    firstFace.rightEyeOpenProbability?.let { rightEyeOpenProb = it }

                                    val originalBitmap = imageProxy.toBitmap()
                                    if (originalBitmap != null) {
                                        val softwareBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)
                                        faceEmbedding = faceNetModel.getFaceEmbedding(softwareBitmap, firstFace.boundingBox)
                                    }

                                    // --- OTOMATISASI UNTUK IDE 1 & 2 (REALTIME) ---

                                    // 1. Cek Liveness (jika diminta)
                                    if (livenessState == LivenessState.WAITING_FOR_BLINK) {
                                        if (leftEyeOpenProb < 0.3 && rightEyeOpenProb < 0.3) {
                                            livenessState = LivenessState.BLINK_DETECTED
                                        }
                                    }

                                    // 2. Cek Wizard Pendaftaran (jika sedang berjalan)
                                    val currentEmbedding = faceEmbedding
                                    if (currentEmbedding != null && appState == AppState.IDLE) {

                                        // Variabel helper untuk angle
                                        val isLookingUp = headEulerAngleX > 20
                                        val isLookingDown = headEulerAngleX < -20
                                        val isLookingCenterV = headEulerAngleX > -10 && headEulerAngleX < 10

                                        val isLookingLeft = headEulerAngleY > 30
                                        val isLookingRight = headEulerAngleY < -30
                                        val isLookingCenterH = headEulerAngleY > -10 && headEulerAngleY < 10

                                        when (enrollmentState) {
                                            // Baris Tengah
                                            EnrollmentState.GET_FRONT_CENTER -> {
                                                if (isLookingCenterV && isLookingCenterH) {
                                                    enrollmentState = EnrollmentState.SENDING_FRONT_CENTER
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_FRONT_LEFT)
                                                }
                                            }
                                            EnrollmentState.GET_FRONT_LEFT -> {
                                                if (isLookingCenterV && isLookingLeft) {
                                                    enrollmentState = EnrollmentState.SENDING_FRONT_LEFT
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_FRONT_RIGHT)
                                                }
                                            }
                                            EnrollmentState.GET_FRONT_RIGHT -> {
                                                if (isLookingCenterV && isLookingRight) {
                                                    enrollmentState = EnrollmentState.SENDING_FRONT_RIGHT
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_TOP_CENTER) // Lanjut ke baris atas
                                                }
                                            }

                                            // Baris Atas
                                            EnrollmentState.GET_TOP_CENTER -> {
                                                if (isLookingUp && isLookingCenterH) {
                                                    enrollmentState = EnrollmentState.SENDING_TOP_CENTER
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_TOP_LEFT)
                                                }
                                            }
                                            EnrollmentState.GET_TOP_LEFT -> {
                                                if (isLookingUp && isLookingLeft) {
                                                    enrollmentState = EnrollmentState.SENDING_TOP_LEFT
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_TOP_RIGHT)
                                                }
                                            }
                                            EnrollmentState.GET_TOP_RIGHT -> {
                                                if (isLookingUp && isLookingRight) {
                                                    enrollmentState = EnrollmentState.SENDING_TOP_RIGHT
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_BOTTOM_CENTER) // Lanjut ke baris bawah
                                                }
                                            }

                                            // Baris Bawah
                                            EnrollmentState.GET_BOTTOM_CENTER -> {
                                                if (isLookingDown && isLookingCenterH) {
                                                    enrollmentState = EnrollmentState.SENDING_BOTTOM_CENTER
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_BOTTOM_LEFT)
                                                }
                                            }
                                            EnrollmentState.GET_BOTTOM_LEFT -> {
                                                if (isLookingDown && isLookingLeft) {
                                                    enrollmentState = EnrollmentState.SENDING_BOTTOM_LEFT
                                                    registerAngle(currentEmbedding, EnrollmentState.GET_BOTTOM_RIGHT)
                                                }
                                            }
                                            EnrollmentState.GET_BOTTOM_RIGHT -> {
                                                if (isLookingDown && isLookingRight) {
                                                    enrollmentState = EnrollmentState.SENDING_BOTTOM_RIGHT
                                                    registerAngle(currentEmbedding, EnrollmentState.FINISHED) // SELESAI
                                                }
                                            }
                                            else -> {} // State lain diabaikan
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

            Box(modifier = Modifier.fillMaxWidth().height(400.dp)) {
                CameraView(modifier = Modifier.fillMaxSize(), analyzer = imageAnalyzer, lifecycleOwner = lifecycleOwner)
                FaceOverlay(modifier = Modifier.fillMaxSize(), faces = detectedFaces, imageSize = imageSize)
                Text(
                    text = if (faceEmbedding != null) "Wajah Terdeteksi" else "Arahkan ke Wajah",
                    color = Color.White, modifier = Modifier.align(Alignment.TopCenter).padding(8.dp)
                )
            }

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
                    // ======================
                    // === TAB PRESENSI (Ide 1) ===
                    // ======================
                    0 -> Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {

                        if (livenessState == LivenessState.WAITING_FOR_BLINK) {
                            Text("Tahan... SEKARANG KEDIPKAN MATA ANDA",
                                color = Color.Blue, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
                        } else {
                            Text("Arahkan wajah Anda ke kamera dan tekan tombol di bawah untuk melakukan presensi.")
                        }

                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = {
                                val embeddingAtClick = faceEmbedding
                                if (embeddingAtClick == null) {
                                    currentMessage = "Error: Wajah hilang. Coba tekan lagi."
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

                    // ===================
                    // === TAB DAFTAR (Ide 2 - 9 Angle) ===
                    // ===================
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
                                    enrollmentState = EnrollmentState.GET_FRONT_CENTER // Mulai wizard dari langkah pertama
                                    currentMessage = ""
                                }
                            },
                            enabled = appState == AppState.IDLE && enrollmentState == EnrollmentState.IDLE && faceEmbedding != null
                        ) {
                            Text("Mulai Pendaftaran Terpandu (9 Angle)")
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        // --- UI WIZARD & Instruksi ---
                        val instructionColor = Color.Blue

                        // Tampilkan pesan progres (jika ada)
                        if (enrollmentMessage.isNotBlank()) {
                            Text(enrollmentMessage, modifier = Modifier.padding(top = 8.dp))
                        }

                        when (enrollmentState) {
                            EnrollmentState.IDLE -> {
                                Text("Isi data di atas lalu tekan 'Mulai' untuk mendaftarkan 9 angle wajah.")
                            }

                            // Baris Tengah
                            EnrollmentState.GET_FRONT_CENTER, EnrollmentState.SENDING_FRONT_CENTER -> {
                                Text("1/9: Lihat LURUS ke kamera...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_FRONT_LEFT, EnrollmentState.SENDING_FRONT_LEFT -> {
                                Text("2/9: Lihat LURUS, lalu TENGOK KE KIRI...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_FRONT_RIGHT, EnrollmentState.SENDING_FRONT_RIGHT -> {
                                Text("3/9: Lihat LURUS, lalu TENGOK KE KANAN...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }

                            // Baris Atas
                            EnrollmentState.GET_TOP_CENTER, EnrollmentState.SENDING_TOP_CENTER -> {
                                Text("4/9: DONGAKKAN KEPALA ke atas...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_TOP_LEFT, EnrollmentState.SENDING_TOP_LEFT -> {
                                Text("5/9: DONGAK & TENGOK KE KIRI...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_TOP_RIGHT, EnrollmentState.SENDING_TOP_RIGHT -> {
                                Text("6/9: DONGAK & TENGOK KE KANAN...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }

                            // Baris Bawah
                            EnrollmentState.GET_BOTTOM_CENTER, EnrollmentState.SENDING_BOTTOM_CENTER -> {
                                Text("7/9: TUNDUKKAN KEPALA ke bawah...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_BOTTOM_LEFT, EnrollmentState.SENDING_BOTTOM_LEFT -> {
                                Text("8/9: TUNDUK & TENGOK KE KIRI...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }
                            EnrollmentState.GET_BOTTOM_RIGHT, EnrollmentState.SENDING_BOTTOM_RIGHT -> {
                                Text("9/9: TUNDUK & TENGOK KE KANAN...", fontWeight = FontWeight.Bold, color = instructionColor)
                            }

                            EnrollmentState.FINISHED -> {
                                Text("Pendaftaran Selesai! 9 angle telah tersimpan.", fontWeight = FontWeight.Bold, color = Color(0xFF4CAF50))
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