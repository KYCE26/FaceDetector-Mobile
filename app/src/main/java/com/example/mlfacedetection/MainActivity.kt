@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalGetImage::class)

package com.example.mlfacedetection

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.ExperimentalAnimationApi
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Fingerprint
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.PersonAdd
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
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

// --- State Navigasi ---
enum class Screen { HOME, PRESENCE, REGISTER }

// --- State Global ---
enum class AppState { IDLE, LOADING }
enum class LivenessState { IDLE, WAITING_FOR_BLINK, BLINK_DETECTED }
enum class EnrollmentState {
    IDLE, // Input Form Mode
    GET_CENTER_NEUTRAL, SENDING_CENTER_NEUTRAL,
    GET_CENTER_SMILE, SENDING_CENTER_SMILE,
    GET_LEFT, SENDING_LEFT, GET_RIGHT, SENDING_RIGHT,
    FINISHED
}

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) { setContent { MLFaceDetectionTheme { AppNavigation(cameraExecutor) } }
        } else { setContent { MLFaceDetectionTheme { PermissionDeniedScreen() } } }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) -> {
                setContent { MLFaceDetectionTheme { AppNavigation(cameraExecutor) } }
            }
            else -> requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

// ==========================================
// 1. NAVIGASI UTAMA (MODERN TRANSITION)
// ==========================================
@OptIn(ExperimentalAnimationApi::class)
@Composable
fun AppNavigation(cameraExecutor: ExecutorService) {
    var currentScreen by remember { mutableStateOf(Screen.HOME) }

    AnimatedContent(
        targetState = currentScreen,
        transitionSpec = {
            fadeIn(animationSpec = tween(300)) togetherWith fadeOut(animationSpec = tween(300))
        }, label = "ScreenTransition"
    ) { screen ->
        when (screen) {
            Screen.HOME -> HomeScreen(
                onNavigateToPresence = { currentScreen = Screen.PRESENCE },
                onNavigateToRegister = { currentScreen = Screen.REGISTER }
            )
            Screen.PRESENCE -> PresenceFeatureScreen(
                cameraExecutor = cameraExecutor,
                onBack = { currentScreen = Screen.HOME }
            )
            Screen.REGISTER -> RegisterFeatureScreen(
                cameraExecutor = cameraExecutor,
                onBack = { currentScreen = Screen.HOME }
            )
        }
    }
}

// ==========================================
// 2. HOME SCREEN (DASHBOARD)
// ==========================================
@Composable
fun HomeScreen(onNavigateToPresence: () -> Unit, onNavigateToRegister: () -> Unit) {
    Scaffold(containerColor = Color(0xFFF5F7FA)) { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(40.dp))
            Icon(Icons.Default.Face, null, Modifier.size(64.dp), tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.height(16.dp))
            Text("Sistem Presensi AI", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("Pilih menu operasional", color = Color.Gray)
            Spacer(modifier = Modifier.height(48.dp))

            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                MenuCard("Presensi", Icons.Default.Fingerprint, Color(0xFF4CAF50), Modifier.weight(1f), onNavigateToPresence)
                MenuCard("Daftar Wajah", Icons.Default.PersonAdd, Color(0xFF2196F3), Modifier.weight(1f), onNavigateToRegister)
            }
            Spacer(modifier = Modifier.weight(1f))
            Text("Versi 4.0 - UI Enhanced", fontSize = 12.sp, color = Color.LightGray)
        }
    }
}

@Composable
fun MenuCard(title: String, icon: ImageVector, color: Color, modifier: Modifier, onClick: () -> Unit) {
    Card(
        modifier = modifier.height(160.dp).clickable { onClick() },
        elevation = CardDefaults.cardElevation(4.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Box(Modifier.size(60.dp).background(color.copy(0.1f), CircleShape), contentAlignment = Alignment.Center) {
                Icon(icon, null, tint = color, modifier = Modifier.size(32.dp))
            }
            Spacer(modifier = Modifier.height(16.dp))
            Text(title, fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
        }
    }
}

// ==========================================
// 3. SCREEN PRESENSI (FULL SCREEN)
// ==========================================
@Composable
fun PresenceFeatureScreen(cameraExecutor: ExecutorService, onBack: () -> Unit) {
    BackHandler { onBack() }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    var detectedFaces by remember { mutableStateOf<List<Face>>(emptyList()) }
    var faceEmbedding by remember { mutableStateOf<FloatArray?>(null) }
    var imageSize by remember { mutableStateOf(Size(0f, 0f)) }

    var appState by remember { mutableStateOf(AppState.IDLE) }
    var livenessState by remember { mutableStateOf(LivenessState.IDLE) }
    var currentMessage by remember { mutableStateOf("Posisikan wajah di dalam oval") }
    var isFlashOn by remember { mutableStateOf(false) }
    var presensiResult by remember { mutableStateOf<RecognizeResponse?>(null) }
    var leftEyeOpenProb by remember { mutableStateOf(1f) }
    var rightEyeOpenProb by remember { mutableStateOf(1f) }

    val faceNetModel = remember { FaceNetModel(context) }
    val faceDetector = remember {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()
        FaceDetection.getClient(options)
    }

    val imageAnalyzer = remember {
        ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build().also {
            it.setAnalyzer(cameraExecutor) { imageProxy ->
                val mediaImage = imageProxy.image
                if (mediaImage != null) {
                    val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                    imageSize = Size(imageProxy.width.toFloat(), imageProxy.height.toFloat())
                    faceDetector.process(image).addOnSuccessListener { faces ->
                        detectedFaces = faces
                        if (faces.isNotEmpty()) {
                            val face = faces.first()
                            face.leftEyeOpenProbability?.let { leftEyeOpenProb = it }
                            face.rightEyeOpenProbability?.let { rightEyeOpenProb = it }
                            val bmp = imageProxy.toBitmap()
                            if (bmp != null) {
                                faceEmbedding = faceNetModel.getFaceEmbedding(bmp.copy(Bitmap.Config.ARGB_8888, true), face.boundingBox)
                            }

                            if (livenessState == LivenessState.WAITING_FOR_BLINK) {
                                if (leftEyeOpenProb < 0.4 && rightEyeOpenProb < 0.4) livenessState = LivenessState.BLINK_DETECTED
                            }
                        } else { faceEmbedding = null }
                    }.addOnCompleteListener { imageProxy.close() }
                } else { imageProxy.close() }
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        CameraView(modifier = Modifier.fillMaxSize(), analyzer = imageAnalyzer, lifecycleOwner = lifecycleOwner)
        FlashAndOvalOverlay(imageSize = imageSize, isFlashOn = isFlashOn)

        // Header
        Row(Modifier.fillMaxWidth().padding(top = 48.dp, start = 16.dp)) {
            IconButton(onClick = onBack, modifier = Modifier.background(Color.Black.copy(0.5f), CircleShape)) {
                Icon(Icons.Default.ArrowBack, null, tint = Color.White)
            }
        }

        // Floating Status
        if (presensiResult == null) {
            Card(
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 100.dp).widthIn(max = 300.dp),
                colors = CardDefaults.cardColors(containerColor = Color.Black.copy(0.7f)),
                shape = RoundedCornerShape(24.dp)
            ) {
                Text(currentMessage, color = Color.White, textAlign = TextAlign.Center, modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp))
            }

            Box(Modifier.align(Alignment.BottomCenter).padding(bottom = 48.dp, start = 24.dp, end = 24.dp)) {
                Button(
                    onClick = {
                        val snapshot = faceEmbedding
                        if (snapshot == null) { currentMessage = "Wajah tidak terdeteksi!"; return@Button }

                        currentMessage = "Tahan... KEDIPKAN MATA SEKARANG!"
                        livenessState = LivenessState.WAITING_FOR_BLINK
                        appState = AppState.LOADING
                        isFlashOn = true

                        coroutineScope.launch {
                            var timeout = 0
                            while (livenessState != LivenessState.BLINK_DETECTED && timeout < 80) { delay(50); timeout++ }
                            isFlashOn = false
                            if (livenessState == LivenessState.BLINK_DETECTED) {
                                currentMessage = "Verifikasi..."
                                try {
                                    val res = ApiService.recognizeFace(snapshot.toList())
                                    presensiResult = res
                                } catch (e: ClientRequestException) {
                                    currentMessage = if(e.response.status == HttpStatusCode.NotFound) "Gagal: Wajah tidak dikenali" else "Error Server"
                                    appState = AppState.IDLE
                                } catch (e: Exception) {
                                    currentMessage = "Error Koneksi"
                                    appState = AppState.IDLE
                                }
                            } else {
                                currentMessage = "Gagal: Tidak ada kedipan mata."
                                appState = AppState.IDLE
                            }
                            livenessState = LivenessState.IDLE
                        }
                    },
                    enabled = faceEmbedding != null && appState == AppState.IDLE,
                    modifier = Modifier.fillMaxWidth().height(56.dp)
                ) {
                    if (appState == AppState.LOADING) CircularProgressIndicator(modifier = Modifier.size(24.dp), color = Color.Black)
                    else { Icon(Icons.Default.CameraAlt, null); Spacer(Modifier.width(8.dp)); Text("Mulai Presensi") }
                }
            }
        }

        if (presensiResult != null) {
            Box(Modifier.fillMaxSize().background(Color.Black.copy(0.8f)), contentAlignment = Alignment.Center) {
                Card(Modifier.padding(32.dp).fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White)) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(32.dp)) {
                        Icon(Icons.Default.CheckCircle, null, tint = Color(0xFF4CAF50), modifier = Modifier.size(64.dp))
                        Text("Berhasil!", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Halo, ${presensiResult!!.name}", style = MaterialTheme.typography.titleMedium)
                        Text("ID: ${presensiResult!!.user_id}", color = Color.Gray)
                        Spacer(modifier = Modifier.height(24.dp))
                        Text("${"%.1f".format(presensiResult!!.similarity * 100)}%", fontSize = 48.sp, fontWeight = FontWeight.Bold, color = Color(0xFF2196F3))
                        Text("Akurasi", color = Color.Gray)
                        Spacer(modifier = Modifier.height(24.dp))
                        Button(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Selesai") }
                    }
                }
            }
        }
    }
}

// ==========================================
// 4. SCREEN DAFTAR (FORM DULU -> BARU KAMERA)
// ==========================================
@Composable
fun RegisterFeatureScreen(cameraExecutor: ExecutorService, onBack: () -> Unit) {
    BackHandler { onBack() }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    // Logic States
    var enrollmentState by remember { mutableStateOf(EnrollmentState.IDLE) } // IDLE = Form Input, Lainnya = Kamera
    var currentMessage by remember { mutableStateOf("") }
    var regUserId by remember { mutableStateOf("") }
    var regUserName by remember { mutableStateOf("") }

    // Face Data
    var faceEmbedding by remember { mutableStateOf<FloatArray?>(null) }
    var headEulerAngleY by remember { mutableStateOf(0f) }
    var smileProb by remember { mutableStateOf(0f) }
    var imageSize by remember { mutableStateOf(Size(0f, 0f)) }

    val faceNetModel = remember { FaceNetModel(context) }
    val faceDetector = remember {
        val options = FaceDetectorOptions.Builder().setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST).setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL).setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL).build()
        FaceDetection.getClient(options)
    }

    val registerAngle: (FloatArray, EnrollmentState) -> Unit = { embedding, nextState ->
        currentMessage = "Mengirim data..."
        coroutineScope.launch {
            try {
                ApiService.registerFace(regUserId, regUserName, embedding.toList())
                enrollmentState = nextState
                currentMessage = ""
            } catch (e: Exception) {
                currentMessage = "Gagal kirim. Ulangi langkah ini."
                enrollmentState = EnrollmentState.values()[enrollmentState.ordinal - 1]
            }
        }
    }

    val imageAnalyzer = remember {
        ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build().also {
            it.setAnalyzer(cameraExecutor) { imageProxy ->
                // HANYA PROSES GAMBAR JIKA SUDAH MASUK MODE KAMERA (Hemat Baterai)
                if (enrollmentState == EnrollmentState.IDLE) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                val mediaImage = imageProxy.image
                if (mediaImage != null) {
                    val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                    imageSize = Size(imageProxy.width.toFloat(), imageProxy.height.toFloat())
                    faceDetector.process(image).addOnSuccessListener { faces ->
                        if (faces.isNotEmpty()) {
                            val face = faces.first()
                            headEulerAngleY = face.headEulerAngleY
                            face.smilingProbability?.let { smileProb = it }
                            val bmp = imageProxy.toBitmap()
                            if (bmp != null) {
                                faceEmbedding = faceNetModel.getFaceEmbedding(bmp.copy(Bitmap.Config.ARGB_8888, true), face.boundingBox)
                            }

                            // Enrollment Logic
                            val curEmb = faceEmbedding
                            if (curEmb != null && enrollmentState != EnrollmentState.IDLE && enrollmentState != EnrollmentState.FINISHED) {
                                val isCenter = headEulerAngleY > -10 && headEulerAngleY < 10
                                val isLeft = headEulerAngleY > 30
                                val isRight = headEulerAngleY < -30
                                val isSmiling = smileProb > 0.6

                                when(enrollmentState) {
                                    EnrollmentState.GET_CENTER_NEUTRAL -> if(isCenter && !isSmiling) {
                                        enrollmentState = EnrollmentState.SENDING_CENTER_NEUTRAL
                                        registerAngle(curEmb, EnrollmentState.GET_CENTER_SMILE)
                                    }
                                    EnrollmentState.GET_CENTER_SMILE -> if(isCenter && isSmiling) {
                                        enrollmentState = EnrollmentState.SENDING_CENTER_SMILE
                                        registerAngle(curEmb, EnrollmentState.GET_LEFT)
                                    }
                                    EnrollmentState.GET_LEFT -> if(isLeft) {
                                        enrollmentState = EnrollmentState.SENDING_LEFT
                                        registerAngle(curEmb, EnrollmentState.GET_RIGHT)
                                    }
                                    EnrollmentState.GET_RIGHT -> if(isRight) {
                                        enrollmentState = EnrollmentState.SENDING_RIGHT
                                        registerAngle(curEmb, EnrollmentState.FINISHED)
                                    }
                                    else -> {}
                                }
                            }
                        } else { faceEmbedding = null }
                    }.addOnCompleteListener { imageProxy.close() }
                } else { imageProxy.close() }
            }
        }
    }

    // --- TAMPILAN UI BERDASARKAN STATE ---

    if (enrollmentState == EnrollmentState.IDLE) {
        // =================================
        // TAMPILAN 1: FORM INPUT (BERSIH)
        // =================================
        Scaffold(
            topBar = {
                CenterAlignedTopAppBar(
                    title = { Text("Pendaftaran User Baru") },
                    navigationIcon = {
                        IconButton(onClick = onBack) { Icon(Icons.Default.ArrowBack, null) }
                    }
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(24.dp))
                Icon(Icons.Default.Badge, null, modifier = Modifier.size(80.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(modifier = Modifier.height(32.dp))

                OutlinedTextField(
                    value = regUserId, onValueChange = { regUserId = it },
                    label = { Text("Nomor ID / NIK") },
                    modifier = Modifier.fillMaxWidth(),
                    leadingIcon = { Icon(Icons.Default.Fingerprint, null) },
                    singleLine = true
                )
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = regUserName, onValueChange = { regUserName = it },
                    label = { Text("Nama Lengkap") },
                    modifier = Modifier.fillMaxWidth(),
                    leadingIcon = { Icon(Icons.Default.Person, null) },
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(48.dp))

                Button(
                    onClick = {
                        if (regUserId.isNotBlank() && regUserName.isNotBlank()) {
                            // PINDAH KE MODE KAMERA
                            enrollmentState = EnrollmentState.GET_CENTER_NEUTRAL
                        } else {
                            currentMessage = "Mohon lengkapi semua data."
                        }
                    },
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text("LANJUT AMBIL FOTO", fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }

                if (currentMessage.isNotBlank()) {
                    Text(currentMessage, color = Color.Red, modifier = Modifier.padding(top = 16.dp))
                }
            }
        }
    } else {
        // =================================
        // TAMPILAN 2: KAMERA FULL SCREEN
        // =================================
        Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
            CameraView(modifier = Modifier.fillMaxSize(), analyzer = imageAnalyzer, lifecycleOwner = lifecycleOwner)

            // Flash & Guide (Flash selalu ON selama proses daftar)
            val isProcessing = enrollmentState != EnrollmentState.FINISHED
            FlashAndOvalOverlay(imageSize = imageSize, isFlashOn = isProcessing)

            // Back Button (Kecil di pojok)
            IconButton(
                onClick = { enrollmentState = EnrollmentState.IDLE }, // Balik ke Form
                modifier = Modifier.padding(top = 48.dp, start = 16.dp).background(Color.Black.copy(0.5f), CircleShape)
            ) {
                Icon(Icons.Default.ArrowBack, null, tint = Color.White)
            }

            // INSTRUKSI ABA-ABA (Besar di Atas)
            if (enrollmentState != EnrollmentState.FINISHED) {
                Card(
                    modifier = Modifier.align(Alignment.TopCenter).padding(top = 100.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.Black.copy(0.7f)),
                    shape = RoundedCornerShape(24.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(24.dp)) {
                        val instructionText = when(enrollmentState) {
                            EnrollmentState.GET_CENTER_NEUTRAL, EnrollmentState.SENDING_CENTER_NEUTRAL -> "1. Lihat Lurus & Wajah Datar"
                            EnrollmentState.GET_CENTER_SMILE, EnrollmentState.SENDING_CENTER_SMILE -> "2. Sekarang SENYUM LEBAR :D"
                            EnrollmentState.GET_LEFT, EnrollmentState.SENDING_LEFT -> "3. Tengok Kiri Pelan-pelan"
                            EnrollmentState.GET_RIGHT, EnrollmentState.SENDING_RIGHT -> "4. Tengok Kanan Pelan-pelan"
                            else -> "Memproses..."
                        }

                        Text(instructionText, color = Color.Yellow, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center)
                        if (currentMessage.isNotBlank()) Text(currentMessage, color = Color.White, fontSize = 14.sp, modifier = Modifier.padding(top = 8.dp))
                    }
                }
            } else {
                // Tampilan Selesai
                Box(modifier = Modifier.fillMaxSize().background(Color.Black.copy(0.8f)), contentAlignment = Alignment.Center) {
                    Card(modifier = Modifier.padding(32.dp), colors = CardDefaults.cardColors(containerColor = Color.White)) {
                        Column(modifier = Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(Icons.Default.CheckCircle, null, tint = Color.Green, modifier = Modifier.size(64.dp))
                            Spacer(modifier = Modifier.height(16.dp))
                            Text("PENDAFTARAN SELESAI!", fontWeight = FontWeight.Bold, fontSize = 20.sp)
                            Spacer(modifier = Modifier.height(24.dp))
                            Button(onClick = {
                                enrollmentState = EnrollmentState.IDLE
                                regUserId = ""
                                regUserName = ""
                            }, modifier = Modifier.fillMaxWidth()) { Text("Tambah User Baru") }
                            Spacer(modifier = Modifier.height(8.dp))
                            OutlinedButton(onClick = onBack, modifier = Modifier.fillMaxWidth()) { Text("Kembali ke Menu Utama") }
                        }
                    }
                }
            }
        }
    }
}

// ==========================================
// HELPER: FLASH & OVAL OVERLAY
// ==========================================
@Composable
fun FlashAndOvalOverlay(imageSize: Size, isFlashOn: Boolean) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val ovalWidth = size.width * 0.65f
        val ovalHeight = ovalWidth * 1.3f
        val ovalLeft = (size.width - ovalWidth) / 2
        val ovalTop = (size.height - ovalHeight) / 2
        val ovalRect = Rect(ovalLeft, ovalTop, ovalLeft + ovalWidth, ovalTop + ovalHeight)

        // Flash Logic:
        // Saat Presensi/Daftar Aktif -> PUTIH (100%) untuk pencahayaan
        // Saat Idle/Fokus -> HITAM Transparan
        val overlayColor = if (isFlashOn) Color.White else Color.Black.copy(alpha = 0.6f)

        val path = Path().apply {
            addRect(Rect(0f, 0f, size.width, size.height))
            addOval(ovalRect)
            fillType = PathFillType.EvenOdd
        }

        drawPath(path, overlayColor)

        drawOval(
            color = if(isFlashOn) Color.Blue else Color.White,
            topLeft = Offset(ovalLeft, ovalTop),
            size = Size(ovalWidth, ovalHeight),
            style = Stroke(width = 4.dp.toPx())
        )
    }
}

@Composable
fun CameraView(modifier: Modifier = Modifier, analyzer: ImageAnalysis, lifecycleOwner: androidx.lifecycle.LifecycleOwner) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember { PreviewView(context) }
    AndroidView(factory = { previewView }, modifier = modifier) {
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analyzer)
            } catch (exc: Exception) { Log.e("CameraView", "Gagal", exc) }
        }, ContextCompat.getMainExecutor(context))
    }
}

@Composable
fun PermissionDeniedScreen() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Butuh izin kamera.", textAlign = TextAlign.Center)
    }
}