# Simple Examples - No Build Tools Required

These examples use only Java/Kotlin standard library with zero external dependencies. Perfect for quick experimentation and learning.

## Java Examples

### Prerequisites
- Java 11 or higher

### Compile and Run

#### Ollama Client
```bash
# Compile
javac GettingStartedOllama.java

# Run
java GettingStartedOllama
```

#### LM Studio Client
```bash
# Compile
javac GettingStartedLmStudio.java

# Run
java GettingStartedLmStudio
```

### One-liner Commands
```bash
# Ollama
javac GettingStartedOllama.java && java GettingStartedOllama

# LM Studio
javac GettingStartedLmStudio.java && java GettingStartedLmStudio
```

## Kotlin Examples

### Prerequisites
- Kotlin compiler (install with `brew install kotlin` on macOS)

### Compile and Run

#### Ollama Client
```bash
# Compile to JAR
kotlinc GettingStartedOllama.kt -include-runtime -d ollama.jar

# Run
java -jar ollama.jar
```

#### LM Studio Client
```bash
# Compile to JAR
kotlinc GettingStartedLmStudio.kt -include-runtime -d lmstudio.jar

# Run
java -jar lmstudio.jar
```

### One-liner Commands
```bash
# Ollama
kotlinc GettingStartedOllama.kt -include-runtime -d ollama.jar && java -jar ollama.jar

# LM Studio
kotlinc GettingStartedLmStudio.kt -include-runtime -d lmstudio.jar && java -jar lmstudio.jar
```

## What These Examples Do

### GettingStartedOllama
- Connects to Ollama running on `localhost:11434`
- Sends a simple "Hello" prompt to the `qwen3` model
- Uses Java's `HttpClient` (Java 11+) for HTTP requests
- Manually parses JSON response (no external libraries)
- Prints the model's response

### GettingStartedLmStudio
- Connects to LM Studio running on `localhost:1234`
- Sends a chat message "Hello" to the `qwen3-4b-mlx` model
- Uses Java's `HttpURLConnection` for HTTP requests
- Manually parses JSON response (no external libraries)
- Prints the model's response

## Code Features

- ✅ **Zero dependencies** - uses only Java/Kotlin standard library
- ✅ **Simple and readable** - easy to understand and modify
- ✅ **Error handling** - catches and reports connection issues
- ✅ **Manual JSON parsing** - no external libraries needed
- ✅ **Timeout handling** - won't hang on failed connections

## Customizing the Examples

### Change the Model

#### Ollama
```java
String jsonString = "{\"model\":\"your-model-name\",\"prompt\":\"Hello\",\"stream\":false,\"Think\":false}";
```

#### LM Studio
```java
String jsonString = "{\"model\":\"your-model-name\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";
```

### Change the Prompt

#### Ollama
```java
String jsonString = "{\"model\":\"qwen3\",\"prompt\":\"Your prompt here\",\"stream\":false,\"Think\":false}";
```

#### LM Studio
```java
String jsonString = "{\"model\":\"qwen3-4b-mlx\",\"messages\":[{\"role\":\"user\",\"content\":\"Your message here\"}]}";
```

### Change the Server URL

#### Ollama
```java
URI uri = URI.create("http://your-server:11434/api/generate");
```

#### LM Studio
```java
URI uri = URI.create("http://your-server:1234/v1/chat/completions");
```

## Troubleshooting

### Java Version Issues
```bash
# Check your Java version
java -version

# If using Java 8-10, upgrade to Java 11+ for HttpClient support
```

### Kotlin Not Found
```bash
# macOS
brew install kotlin

# Linux
sudo snap install kotlin --classic

# Windows
choco install kotlin
```

### Connection Refused
```bash
# Ensure Ollama is running
ollama serve

# Ensure LM Studio server is started in the UI
```

### Model Not Found
```bash
# For Ollama, pull the model first
ollama pull qwen3

# For LM Studio, download and load model in the UI
```

## Running in an IDE

Prefer to use IntelliJ IDEA, Eclipse, VS Code, or NetBeans?

See **[../IDE_SETUP.md](../IDE_SETUP.md)** for complete instructions on:
- Opening these files in your IDE
- Running with a single click
- Setting up run configurations
- Debugging your code

## Next Steps

Once you're comfortable with these simple examples, consider moving to the Maven project in `../maven-example/` for:
- Proper dependency management
- Better JSON parsing with Gson
- Modern HTTP client with OkHttp
- Production-ready code structure
- Easier testing and maintenance
