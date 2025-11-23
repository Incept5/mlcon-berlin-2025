# GenAI Clients - Gradle Example

Production-ready Java clients for Ollama and LM Studio using modern HTTP libraries and proper dependency management with Gradle.

## Features

- ✅ **Modern HTTP client** - Uses OkHttp for efficient networking
- ✅ **Proper JSON handling** - Gson for type-safe JSON parsing
- ✅ **Gradle tasks** - Easy switching between Ollama and LM Studio
- ✅ **Clean code structure** - Follows Java best practices
- ✅ **Resource management** - Proper cleanup of HTTP connections
- ✅ **Error handling** - Comprehensive exception handling
- ✅ **No deprecation warnings** - Uses current Java APIs
- ✅ **Kotlin DSL** - Modern Gradle configuration using Kotlin

## Prerequisites

- **Java 17+** (Required for modern features)
- **Gradle 8.0+** (Build tool) - Wrapper included
- **Ollama** running on `localhost:11434` (for Ollama client)
- **LM Studio** running on `localhost:1234` (for LM Studio client)

## Quick Start

### Option 1: Command Line

#### 1. Build the Project
```bash
./gradlew build
```

#### 2. Run the Clients

**Using Custom Gradle Tasks (Recommended)**
```bash
# Run Ollama client
./gradlew runOllama

# Run LM Studio client
./gradlew runLmStudio

# Run default client (Ollama)
./gradlew run
```

**Using Direct Class Execution**
```bash
# Run Ollama client
./gradlew run --args="com.example.OllamaClient"

# Run LM Studio client
./gradlew run --args="com.example.LmStudioClient"
```

#### 3. Package as Executable JAR
```bash
# Build executable JAR
./gradlew clean jar

# Run the JAR (default: OllamaClient)
java -jar build/libs/gradle-example-1.0.0.jar
```

### Option 2: Using an IDE

Prefer IntelliJ IDEA, Eclipse, VS Code, or NetBeans?

**See [../IDE_SETUP.md](../IDE_SETUP.md)** for complete step-by-step instructions on:
- Importing the Gradle project into your IDE
- Setting up run configurations
- Running with a single click
- Debugging your code
- IDE-specific tips and shortcuts

**Quick Summary for IntelliJ IDEA:**
1. File → Open → Select `build.gradle.kts`
2. Wait for Gradle sync to complete
3. Open `src/main/java/com/example/OllamaClient.java`
4. Right-click → Run 'OllamaClient.main()'

## Project Structure

```
gradle-example/
├── build.gradle.kts                  # Gradle configuration (Kotlin DSL)
├── settings.gradle.kts               # Project settings
├── gradle/                           # Gradle wrapper files
│   └── wrapper/
├── gradlew                           # Gradle wrapper script (Unix)
├── gradlew.bat                       # Gradle wrapper script (Windows)
├── README.md                         # This file
└── src/
    └── main/
        └── java/
            └── com/
                └── example/
                    ├── OllamaClient.java      # Ollama client implementation
                    └── LmStudioClient.java    # LM Studio client implementation
```

## Dependencies

This project uses two main dependencies:

### OkHttp (v4.12.0)
- Modern, efficient HTTP client
- Connection pooling and keep-alive
- Automatic retry and redirect handling
- Better performance than legacy HttpURLConnection

### Gson (v2.10.1)
- Type-safe JSON parsing and generation
- Intuitive API for working with JSON
- Better than manual string parsing
- Widely used in production applications

## Code Examples

### OllamaClient.java

Connects to Ollama and generates text:

```java
// Build request JSON
JsonObject requestJson = new JsonObject();
requestJson.addProperty("model", "qwen3");
requestJson.addProperty("prompt", "Hello");
requestJson.addProperty("stream", false);

// Make HTTP POST request
RequestBody body = RequestBody.create(requestJson.toString(), JSON);
Request request = new Request.Builder()
    .url("http://localhost:11434/api/generate")
    .post(body)
    .build();

// Parse response
JsonObject jsonResponse = JsonParser.parseString(responseBody).getAsJsonObject();
String content = jsonResponse.get("response").getAsString();
```

### LmStudioClient.java

Connects to LM Studio for chat completions:

```java
// Build chat message
JsonObject message = new JsonObject();
message.addProperty("role", "user");
message.addProperty("content", "Hello");

JsonArray messages = new JsonArray();
messages.add(message);

JsonObject requestJson = new JsonObject();
requestJson.addProperty("model", "qwen3-4b-mlx");
requestJson.add("messages", messages);

// Extract response from chat completion
String content = jsonResponse
    .getAsJsonArray("choices")
    .get(0).getAsJsonObject()
    .getAsJsonObject("message")
    .get("content").getAsString();
```

## Customization

### Change the Model

Edit the model name in the source files:

**OllamaClient.java:**
```java
requestJson.addProperty("model", "your-model-name");
```

**LmStudioClient.java:**
```java
requestJson.addProperty("model", "your-model-name");
```

### Change the Prompt

**OllamaClient.java:**
```java
requestJson.addProperty("prompt", "Your prompt here");
```

**LmStudioClient.java:**
```java
message.addProperty("content", "Your message here");
```

### Change Server URLs

**OllamaClient.java:**
```java
private static final String OLLAMA_URL = "http://your-server:11434/api/generate";
```

**LmStudioClient.java:**
```java
private static final String LM_STUDIO_URL = "http://your-server:1234/v1/chat/completions";
```

### Add More Options

Both clients support additional parameters:

**Ollama:**
```java
requestJson.addProperty("temperature", 0.8);
requestJson.addProperty("top_p", 0.9);
requestJson.addProperty("max_tokens", 100);
```

**LM Studio:**
```java
requestJson.addProperty("temperature", 0.8);
requestJson.addProperty("max_tokens", 100);
requestJson.addProperty("top_p", 0.9);
```

## Gradle Commands Reference

```bash
# Clean build artifacts
./gradlew clean

# Compile source code
./gradlew compileJava

# Run tests (when added)
./gradlew test

# Package as JAR
./gradlew jar

# Build project (compile + test + jar)
./gradlew build

# Run with Ollama client
./gradlew runOllama

# Run with LM Studio client
./gradlew runLmStudio

# Run default client
./gradlew run

# Clean and build in one command
./gradlew clean build

# Skip tests during build
./gradlew clean build -x test

# List all available tasks
./gradlew tasks

# Show project dependencies
./gradlew dependencies

# Refresh dependencies
./gradlew build --refresh-dependencies
```

## Gradle Wrapper

This project includes the Gradle Wrapper, which ensures everyone uses the same Gradle version:

```bash
# Unix/macOS
./gradlew [task]

# Windows
gradlew.bat [task]
```

The wrapper automatically downloads the correct Gradle version on first use.

## Known Warnings

You may see some warnings when running the application:

### OkHttp Thread Warnings
```
[WARNING] thread Thread[#XX,OkHttp TaskRunner,5,...] will linger despite being 
asked to die via interruption
```
**This is expected behavior.** It occurs because Gradle's application plugin forcibly terminates the JVM, interrupting OkHttp's background threads. The code includes proper cleanup in the `finally` block, but Gradle's termination is immediate. This warning is informational only and doesn't affect the HTTP request functionality.

## Troubleshooting

### Build Issues

#### Java Version
```bash
# Check Java version (must be 17+)
java -version

# Set JAVA_HOME if needed (macOS/Linux)
export JAVA_HOME=/path/to/java17

# Set JAVA_HOME (Windows)
set JAVA_HOME=C:\path\to\java17
```

#### Gradle Issues
```bash
# Use the wrapper (recommended)
./gradlew build

# Or install Gradle globally
# macOS
brew install gradle

# Linux
sudo apt install gradle

# Windows
choco install gradle
```

#### Dependency Download Issues
```bash
# Clear Gradle cache
rm -rf ~/.gradle/caches

# Force refresh dependencies
./gradlew build --refresh-dependencies
```

### Runtime Issues

#### Connection Refused
```bash
# Ensure Ollama is running
ollama serve

# Check Ollama is accessible
curl http://localhost:11434/api/generate

# For LM Studio, ensure server is started in the UI
```

#### Model Not Found
```bash
# List available Ollama models
ollama list

# Pull required model
ollama pull qwen3

# For LM Studio, download model in the UI
```

#### Wrong Model Name
- Ollama: Check available models with `ollama list`
- LM Studio: Check model name in the UI's "Server" tab

## Adding to Your Project

To use these clients in your own Gradle project:

**build.gradle.kts:**
```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
}
```

**build.gradle (Groovy DSL):**
```groovy
dependencies {
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    implementation 'com.google.code.gson:gson:2.10.1'
}
```

Then copy the client classes into your project and modify as needed.

## Extending the Clients

### Add Streaming Support

```java
// Enable streaming in request
requestJson.addProperty("stream", true);

// Handle streamed response
try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(response.body().byteStream()))) {
    String line;
    while ((line = reader.readLine()) != null) {
        JsonObject chunk = JsonParser.parseString(line).getAsJsonObject();
        String content = chunk.get("response").getAsString();
        System.out.print(content);
    }
}
```

### Add Conversation History

```java
// For LM Studio (chat-based)
JsonArray messages = new JsonArray();

// Add system message
JsonObject systemMsg = new JsonObject();
systemMsg.addProperty("role", "system");
systemMsg.addProperty("content", "You are a helpful assistant.");
messages.add(systemMsg);

// Add previous conversation
JsonObject prevMsg = new JsonObject();
prevMsg.addProperty("role", "assistant");
prevMsg.addProperty("content", "Previous response");
messages.add(prevMsg);

// Add new user message
JsonObject newMsg = new JsonObject();
newMsg.addProperty("role", "user");
newMsg.addProperty("content", "New question");
messages.add(newMsg);

requestJson.add("messages", messages);
```

### Add Configuration Class

```java
public class GenAIConfig {
    private final String baseUrl;
    private final String model;
    private final double temperature;
    
    public GenAIConfig(String baseUrl, String model, double temperature) {
        this.baseUrl = baseUrl;
        this.model = model;
        this.temperature = temperature;
    }
    
    // Getters...
}
```

## Gradle vs Maven

This project uses **Gradle with Kotlin DSL** for build configuration. Key differences from Maven:

### Advantages of Gradle:
- **Faster builds** - Incremental compilation and build cache
- **More flexible** - Programmatic build scripts
- **Better dependency management** - Dynamic versions and excludes
- **Modern tooling** - Kotlin DSL provides better IDE support
- **Multi-project builds** - Native support for complex projects

### Build File Comparison:

**Gradle (Kotlin DSL):**
```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
}
```

**Maven:**
```xml
<dependencies>
    <dependency>
        <groupId>com.squareup.okhttp3</groupId>
        <artifactId>okhttp</artifactId>
        <version>4.12.0</version>
    </dependency>
</dependencies>
```

### Task Comparison:

| Task | Gradle | Maven |
|------|--------|-------|
| Clean | `./gradlew clean` | `mvn clean` |
| Compile | `./gradlew compileJava` | `mvn compile` |
| Test | `./gradlew test` | `mvn test` |
| Package | `./gradlew jar` | `mvn package` |
| Run | `./gradlew run` | `mvn exec:java` |

## Next Steps

1. **Explore the code** - Read through `OllamaClient.java` and `LmStudioClient.java`
2. **Modify the examples** - Try different models and prompts
3. **Add features** - Implement streaming, conversation history, etc.
4. **Build your app** - Use these clients as building blocks
5. **Add tests** - Create JUnit tests for your implementations
6. **Learn Gradle** - Explore the build.gradle.kts configuration

## Resources

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [OkHttp Documentation](https://square.github.io/okhttp/)
- [Gson Documentation](https://github.com/google/gson)
- [Gradle Documentation](https://docs.gradle.org/)
- [Gradle Kotlin DSL](https://docs.gradle.org/current/userguide/kotlin_dsl.html)

## License

MIT

---
