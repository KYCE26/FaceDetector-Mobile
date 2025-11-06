package com.example.mlfacedetection // Pastikan ini adalah package Anda

import io.ktor.client.*
import io.ktor.client.call.body
import io.ktor.client.engine.android.*
import io.ktor.client.plugins.* // Import
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.* // Import
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json


// =========================================================
// ▼▼▼ PENTING: GANTI DENGAN IP KOMPUTER ANDA ▼▼▼
// =========================================================
private const val BASE_URL = "https://web-production-dff34.up.railway.app" // <-- PASTIKAN INI IP ANDA

// --- Data Class untuk JSON (Request) ---
@Serializable
data class UserRegisterRequest(
    val user_id: String,
    val name: String,
    val embedding: List<Float>
)

@Serializable
data class RecognizeRequest(
    val embedding: List<Float>
)

// --- Data Class untuk JSON (Response) ---

// PERUBAHAN DI SINI: Sesuaikan dengan 'main.py' V3
@Serializable
data class RegisterResponse(
    val status: String,
    val user_id: String,
    val name: String,
    val new_embedding_id: String // Kita menangkap ID embedding baru
)

@Serializable
data class RecognizeResponse(
    val status: String,
    val user_id: String,
    val name: String,
    val similarity: Float
)


// --- Ktor Client ---
object ApiService {

    val client = HttpClient(Android) {
        install(ContentNegotiation) {
            json(Json {
                prettyPrint = true
                isLenient = true
                ignoreUnknownKeys = true
            })
        }

        // Penting: agar 'try-catch' kita bisa menangkap error 404/500
        expectSuccess = true

        // Opsi: Tambahkan timeout agar tidak menunggu selamanya
        // install(HttpTimeout) {
        //    requestTimeoutMillis = 15000 // 15 detik
        // }
    }

    suspend fun registerFace(userId: String, name: String, embedding: List<Float>): RegisterResponse {
        val requestBody = UserRegisterRequest(user_id = userId, name = name, embedding = embedding)

        val response: HttpResponse = client.post("$BASE_URL/register") {
            contentType(ContentType.Application.Json)
            setBody(requestBody)
        }
        return response.body()
    }

    suspend fun recognizeFace(embedding: List<Float>): RecognizeResponse {
        val requestBody = RecognizeRequest(embedding = embedding)

        val response: HttpResponse = client.post("$BASE_URL/recognize") {
            contentType(ContentType.Application.Json)
            setBody(requestBody)
        }
        return response.body()
    }
}