import java.io.*
import java.net.*

fun main() {
    try {
        val jsonString = """{"model":"qwen3-4b-mlx","messages":[{"role":"user","content":"Hello"}]}"""

        val uri = URI.create("http://localhost:1234/v1/chat/completions")
        val connection = uri.toURL().openConnection() as HttpURLConnection

        connection.requestMethod = "POST"
        connection.setRequestProperty("Content-Type", "application/json")
        connection.doOutput = true
        connection.connectTimeout = 10000
        connection.readTimeout = 30000

        connection.outputStream.use { outputStream ->
            OutputStreamWriter(outputStream).use { writer ->
                writer.write(jsonString)
                writer.flush()
            }
        }

        val statusCode = connection.responseCode

        val responseBody = if (statusCode in 200..299) {
            connection.inputStream.use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    reader.readText()
                }
            }
        } else {
            connection.errorStream.use { errorStream ->
                BufferedReader(InputStreamReader(errorStream)).use { reader ->
                    val error = reader.readText()
                    println("Error response: $error")
                    return
                }
            }
        }

        val content = extractMessageContent(responseBody)
        println(content.trim())

    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}

fun extractMessageContent(json: String): String {
    // Look for the content field more flexibly
    val contentStart = json.indexOf("\"content\":")
    if (contentStart == -1) return ""

    // Find the opening quote of the content value
    val quoteStart = json.indexOf("\"", contentStart + "\"content\":".length)
    if (quoteStart == -1) return ""

    val valueStart = quoteStart + 1
    var contentEnd = valueStart

    // Find the end of the content field, accounting for escaped quotes
    while (contentEnd < json.length) {
        val c = json[contentEnd]

        if (c == '\\') {
            // Skip escaped character
            contentEnd += 2
            continue
        }

        if (c == '"') {
            // Found the end of the content string
            break
        }
        contentEnd++
    }

    if (contentEnd >= json.length) return ""

    val content = json.substring(valueStart, contentEnd)
    return content.replace("\\n", "\n").replace("\\\"", "\"").replace("\\t", "\t")
}