import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse

fun main() {
    try {
        val client = HttpClient.newHttpClient()

        val jsonString = """{"model":"qwen3","prompt":"Hello","stream":false,"Think":false}"""

        val request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:11434/api/generate"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonString))
            .build()

        val response = client.send(request, HttpResponse.BodyHandlers.ofString())

        val responseBody = response.body()
        val responseText = extractJsonValue(responseBody, "response")
        println(responseText)

    } catch (e: Exception) {
        e.printStackTrace()
    }
}

fun extractJsonValue(json: String, key: String): String {
    val searchKey = "\"$key\":\""
    val startIndex = json.indexOf(searchKey)
    if (startIndex == -1) return ""

    val valueStart = startIndex + searchKey.length
    val endIndex = json.indexOf("\"", valueStart)
    if (endIndex == -1) return ""

    return json.substring(valueStart, endIndex).replace("\\n", "\n").replace("\\\"", "\"")
}