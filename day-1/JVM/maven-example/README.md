# GenAI Clients - Maven Example

Simple Java clients for Ollama and LM Studio using Maven with HTTP libraries.

## Dependencies

- **OkHttp**: Modern HTTP client for efficient networking
- **Gson**: JSON parsing and serialization

## Prerequisites

- Java 17+
- Maven 3.6+
- Ollama running on `localhost:11434` (for Ollama client)
- LM Studio running on `localhost:1234` (for LM Studio client)

## Build

```bash
mvn compile
```

## Run

### Ollama Client
```bash
mvn exec:java -Pollama
```

### LM Studio Client
```bash
mvn exec:java -Plmstudio
```

### Alternative run commands
```bash
# Run Ollama client directly
mvn exec:java -Dexec.mainClass="com.example.OllamaClient"

# Run LM Studio client directly
mvn exec:java -Dexec.mainClass="com.example.LmStudioClient"
```

## Project Structure

```
maven-example/
├── pom.xml
├── README.md
└── src/main/java/com/example/
    ├── OllamaClient.java
    └── LmStudioClient.java
```

## Features

- ✅ Simple, clean code using popular HTTP libraries
- ✅ Proper JSON parsing with Gson
- ✅ Maven profiles for easy execution
- ✅ Error handling and resource management
- ✅ No deprecation warnings

## Known Warnings

You may see some warnings when running:

### Maven/Java Runtime Warnings
```
WARNING: A restricted method in java.lang.System has been called
WARNING: A terminally deprecated method in sun.misc.Unsafe has been called
```
These are from Maven's internal dependencies and cannot be avoided. They don't affect functionality.

### OkHttp Thread Warnings
```
[WARNING] thread Thread[#XX,OkHttp TaskRunner,5,...] will linger despite being asked to die via interruption
```
These occur because Maven's exec plugin forcibly terminates the JVM, interrupting OkHttp's background threads. This is normal behavior and doesn't affect the functionality of the HTTP requests. The warnings are informational only.