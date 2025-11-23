# Quick Start Guide

Get up and running with GenAI JVM examples in under 5 minutes!

## 🚀 Fastest Path to Success

### Step 1: Check Prerequisites
```bash
./run-examples.sh check
```

This will verify you have Java, and optionally Maven and Kotlin installed.

### Step 2: Start a Server (REQUIRED)

⚠️ **IMPORTANT:** The server MUST be running before executing any examples!

**Option A: Ollama (Recommended)**
```bash
# Start Ollama server
ollama serve

# In another terminal: Pull a model
ollama pull qwen3

# Verify server is running
curl http://localhost:11434/api/tags
```

**Option B: LM Studio**
1. **Open LM Studio application**
2. **Download and load a model** (e.g., qwen3-4b-mlx)
3. **Go to "Local Server" or "Server" tab**
4. **Click "Start Server"** (defaults to localhost:1234)
5. **Verify server is running:**
   ```bash
   curl http://localhost:1234/v1/models
   ```

⚠️ **Without a running server, you'll get:** `Error: Failed to connect to localhost`

### Step 3: Run an Example

**Simplest (Java only, no build tools):**
```bash
./run-examples.sh simple-java-ollama
```

**Or with Maven (recommended for production):**
```bash
./run-examples.sh maven-ollama
```

## 📋 All Available Commands

### Simple Examples (No Build Tools)
```bash
# Java examples
./run-examples.sh simple-java-ollama      # Ollama with Java
./run-examples.sh simple-java-lmstudio    # LM Studio with Java

# Kotlin examples
./run-examples.sh simple-kotlin-ollama    # Ollama with Kotlin
./run-examples.sh simple-kotlin-lmstudio  # LM Studio with Kotlin
```

### Maven Examples (Production-Ready)
```bash
./run-examples.sh maven-ollama           # Ollama with Maven
./run-examples.sh maven-lmstudio         # LM Studio with Maven
./run-examples.sh maven-build            # Build Maven project
./run-examples.sh maven-package          # Create executable JAR
```

### Utilities
```bash
./run-examples.sh check                  # Check prerequisites
./run-examples.sh help                   # Show all options
```

## 🎯 Choose Your Path

### 🏃‍♂️ Quick Experimentation
Use **Simple Examples** if you:
- Want to run something immediately
- Are learning the basics
- Don't want to install build tools
- Need minimal setup

**Start here:** `./run-examples.sh simple-java-ollama`

### 🏗️ Building Production Apps
Use **Maven Examples** if you:
- Need dependency management
- Want production-ready code
- Are building a real application
- Need proper project structure

**Start here:** `./run-examples.sh maven-build && ./run-examples.sh maven-ollama`

## 🔧 Manual Commands

### Simple Java (No Maven)
```bash
cd simple-examples

# Ollama
javac GettingStartedOllama.java && java GettingStartedOllama

# LM Studio
javac GettingStartedLmStudio.java && java GettingStartedLmStudio
```

### Simple Kotlin (No Maven)
```bash
cd simple-examples

# Ollama
kotlinc GettingStartedOllama.kt -include-runtime -d ollama.jar && java -jar ollama.jar

# LM Studio
kotlinc GettingStartedLmStudio.kt -include-runtime -d lmstudio.jar && java -jar lmstudio.jar
```

### Maven Project
```bash
cd maven-example

# Build
mvn clean compile

# Run Ollama
mvn exec:java -Pollama

# Run LM Studio
mvn exec:java -Plmstudio

# Create JAR
mvn clean package
java -jar target/genai-clients-1.0.0.jar
```

## 📁 Project Structure

```
day-1/JVM/
├── run-examples.sh              # 🎯 Start here!
├── QUICK_START.md               # This file
├── README.md                    # Detailed documentation
│
├── simple-examples/             # Zero-dependency examples
│   ├── README.md
│   ├── GettingStartedOllama.java
│   ├── GettingStartedOllama.kt
│   ├── GettingStartedLmStudio.java
│   └── GettingStartedLmStudio.kt
│
└── maven-example/               # Production-ready project
    ├── README.md
    ├── pom.xml
    └── src/main/java/com/example/
        ├── OllamaClient.java
        └── LmStudioClient.java
```

## 🆘 Troubleshooting

### "Connection refused" or "Failed to connect to localhost"
**Problem:** Can't connect to Ollama/LM Studio

**Root Cause:** The server is not running! This is the most common error.

**Solutions:**

**For Ollama:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve

# Pull a model if you haven't already
ollama pull qwen3
```

**For LM Studio:**
1. Open the LM Studio application
2. Go to the "Local Server" or "Server" tab
3. Click "Start Server" button
4. Verify it's running:
   ```bash
   curl http://localhost:1234/v1/models
   ```

⚠️ **Remember:** The server must be running in the background while you execute the examples!

### "Model not found"
**Problem:** Server doesn't have the requested model

**Solutions:**
```bash
# For Ollama - pull the model
ollama pull qwen3
ollama list  # See all models

# For LM Studio - download model in the UI
# Check model name in the "Server" tab
```

### "Command not found: javac"
**Problem:** Java compiler not installed

**Solutions:**
```bash
# Check Java version
java -version  # Must be 11+

# macOS
brew install openjdk@17

# Ubuntu/Debian
sudo apt install openjdk-17-jdk

# Windows
choco install openjdk17
```

### "Command not found: mvn"
**Problem:** Maven not installed

**Solutions:**
```bash
# macOS
brew install maven

# Ubuntu/Debian
sudo apt install maven

# Windows
choco install maven
```

### "Command not found: kotlinc"
**Problem:** Kotlin compiler not installed

**Solutions:**
```bash
# macOS
brew install kotlin

# Ubuntu/Debian
sudo snap install kotlin --classic

# Windows
choco install kotlin
```

## 💡 Tips

### Change the Model
Edit the source file and change the model name:
```java
// For Ollama
"model":"qwen3"  →  "model":"llama2"

// For LM Studio
"model":"qwen3-4b-mlx"  →  "model":"your-model-name"
```

### Change the Prompt
```java
// For Ollama
"prompt":"Hello"  →  "prompt":"Tell me a joke"

// For LM Studio
"content":"Hello"  →  "content":"Tell me a joke"
```

### Use Different Servers
```java
// For Ollama
"http://localhost:11434"  →  "http://your-server:11434"

// For LM Studio
"http://localhost:1234"  →  "http://your-server:1234"
```

## 🎓 Learning Path

1. **Start Simple** → Run `./run-examples.sh simple-java-ollama`
2. **Read the Code** → Open `simple-examples/GettingStartedOllama.java`
3. **Modify and Experiment** → Change the prompt, try different models
4. **Try Kotlin** → Run `./run-examples.sh simple-kotlin-ollama`
5. **Learn Maven** → Run `./run-examples.sh maven-build`
6. **Go Production** → Build your app using the Maven project structure

## 🖥️ Using an IDE?

Prefer to run examples in IntelliJ IDEA, Eclipse, VS Code, or NetBeans?

**See the [IDE_SETUP.md](IDE_SETUP.md) guide** for complete instructions on:
- Opening projects in your IDE
- Configuring run configurations
- Running and debugging examples
- IDE-specific tips and shortcuts

## 📚 Further Reading

- [README.md](README.md) - Complete documentation
- [IDE_SETUP.md](IDE_SETUP.md) - IDE setup guide (IntelliJ, Eclipse, VS Code, NetBeans)
- [simple-examples/README.md](simple-examples/README.md) - Simple examples guide
- [maven-example/README.md](maven-example/README.md) - Maven project guide

## 🐛 Still Having Issues?

1. Run the check command: `./run-examples.sh check`
2. Read the error messages carefully
3. Check the relevant README.md for detailed troubleshooting
4. Ensure your server (Ollama/LM Studio) is actually running
5. Verify the model is downloaded and available

## 🎉 Success!

Once you see the AI's response printed to your terminal, you're ready to start building!

**Next steps:**
- Modify the examples to suit your needs
- Explore the Maven project for production patterns
- Build your own GenAI application
- Have fun experimenting with different models!
