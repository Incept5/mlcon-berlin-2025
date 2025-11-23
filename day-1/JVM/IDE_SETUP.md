# IDE Setup Guide

Complete guide for running the JVM examples in popular IDEs (IntelliJ IDEA, Eclipse, VS Code, and NetBeans).

## Table of Contents
- [IntelliJ IDEA](#intellij-idea)
- [Eclipse](#eclipse)
- [VS Code](#vs-code)
- [NetBeans](#netbeans)

---

## IntelliJ IDEA

### Option 1: Quick Run - Simple Examples (No Build Tool)

This is the fastest way to run the simple examples without any project configuration.

#### Step 1: Open File
1. Open IntelliJ IDEA
2. **File → Open**
3. Navigate to `day-1/JVM/simple-examples/`
4. Select `GettingStartedOllama.java` (or any other .java/.kt file)
5. Click **Open**
6. When prompted, choose **"Open as Project"** or **"This Window"**

#### Step 2: Configure JDK
1. **File → Project Structure** (or press `Cmd+;` on macOS / `Ctrl+Alt+Shift+S` on Windows)
2. Under **Project Settings → Project**:
   - Set **SDK** to Java 11 or higher (Java 17+ recommended)
   - If no JDK is listed, click **+ → Add SDK → Download JDK**
   - Select **Language Level**: 11 or higher
3. Click **OK**

#### Step 3: Run the File
1. Right-click on the file in the editor
2. Select **Run 'GettingStartedOllama.main()'**
3. The output will appear in the **Run** panel at the bottom

**Alternatively:**
- Click the green ▶️ play button in the gutter next to the `main()` method
- Press `Ctrl+Shift+R` (macOS) or `Shift+F10` (Windows/Linux)

#### For Kotlin Files
Same process works for `.kt` files. IntelliJ will automatically detect Kotlin and configure it.

### Option 2: Maven Project (Recommended for Development)

For the production-ready Maven project with proper dependency management.

#### Step 1: Import Maven Project
1. Open IntelliJ IDEA
2. **File → Open**
3. Navigate to `day-1/JVM/maven-example/`
4. Select the `pom.xml` file
5. Click **Open**
6. Choose **"Open as Project"**
7. In the dialog, select **"Open as Maven Project"** or check **"Trust Project"**

#### Step 2: Wait for Maven Sync
- IntelliJ will automatically download dependencies
- Wait for the progress bar at the bottom to complete
- You'll see "Maven: Importing" notification

#### Step 3: Run the Examples

**Method A: Using Run Configurations (Easiest)**

1. Open the Maven tool window: **View → Tool Windows → Maven**
2. Expand `genai-clients → Plugins → exec`
3. Double-click `exec:java` to run (defaults to OllamaClient)

**Method B: Using IntelliJ Run Configurations**

1. **Run → Edit Configurations**
2. Click **+** → **Application**
3. Configure:
   - **Name**: `OllamaClient`
   - **Main class**: `com.example.OllamaClient` (click **...** to browse)
   - **Module**: `genai-clients`
   - **JRE**: Java 17 or higher
4. Click **OK**
5. Select your configuration from the dropdown and click the green ▶️ play button

**Method C: Direct File Execution**

1. Open `src/main/java/com/example/OllamaClient.java`
2. Right-click anywhere in the file
3. Select **Run 'OllamaClient.main()'**

#### Step 4: Create Multiple Run Configurations

Create separate run configurations for each client:

1. **Run → Edit Configurations**
2. Click **+** → **Application**
3. For Ollama:
   - **Name**: `Ollama Client`
   - **Main class**: `com.example.OllamaClient`
4. Click **+** → **Application** again
5. For LM Studio:
   - **Name**: `LM Studio Client`
   - **Main class**: `com.example.LmStudioClient`

Now you can switch between them using the dropdown next to the play button.

### Option 3: Create IntelliJ Project from Scratch

If you want to create a fresh IntelliJ project:

#### For Simple Examples:
1. **File → New → Project**
2. Select **Java** (or **Kotlin**)
3. Set **JDK** to 17+
4. **Build System**: None
5. Click **Next** → **Finish**
6. Copy the simple example files into `src/` directory
7. Right-click file → **Run**

#### For Maven Project:
1. **File → New → Project**
2. Select **Maven**
3. Set **JDK** to 17+
4. Click **Next**
5. Set **GroupId**: `com.example`
6. Set **ArtifactId**: `genai-clients`
7. Click **Finish**
8. Replace the generated `pom.xml` with the one from `maven-example/`
9. Copy the source files to `src/main/java/com/example/`
10. Wait for Maven sync
11. Run as described above

### Keyboard Shortcuts (IntelliJ)

| Action | macOS | Windows/Linux |
|--------|-------|---------------|
| Run | `Ctrl+Shift+R` | `Shift+F10` |
| Run (context) | `Ctrl+Shift+R` | `Shift+F10` |
| Edit Configurations | `Cmd+Shift+A` → type "edit" | `Ctrl+Shift+A` → type "edit" |
| Project Structure | `Cmd+;` | `Ctrl+Alt+Shift+S` |
| Maven Tool Window | `Cmd+3` | `Alt+3` |

### Troubleshooting IntelliJ

#### "Cannot resolve symbol"
- **File → Invalidate Caches → Invalidate and Restart**
- Maven projects: Right-click `pom.xml` → **Maven → Reload Project**

#### "SDK not configured"
- **File → Project Structure → Project**
- Set **SDK** to Java 11+
- If not available: **+ → Download JDK** → Select version 17+

#### Maven dependencies not downloading
- **View → Tool Windows → Maven**
- Click refresh button (🔄) in the Maven tool window
- Or: **File → Invalidate Caches → Invalidate and Restart**

#### "Connection refused" when running
- Ensure Ollama/LM Studio is running before running the code
- Check server URLs in the code match your setup

---

## Eclipse

### Simple Examples

#### Step 1: Create Java Project
1. **File → New → Java Project**
2. **Project name**: `GenAI-Simple-Examples`
3. **JRE**: Use Java 11 or higher
4. **Project layout**: Use default
5. Click **Finish**

#### Step 2: Add Source Files
1. Right-click project → **New → Class**
2. **Name**: `GettingStartedOllama`
3. Check **public static void main(String[] args)**
4. Click **Finish**
5. Copy the code from `simple-examples/GettingStartedOllama.java`
6. Paste into the editor, replacing the generated code

#### Step 3: Run
1. Right-click the file in **Package Explorer**
2. **Run As → Java Application**
3. Output appears in **Console** view

### Maven Project

#### Step 1: Import Maven Project
1. **File → Import**
2. **Maven → Existing Maven Projects**
3. Click **Next**
4. **Root Directory**: Browse to `day-1/JVM/maven-example/`
5. Ensure `pom.xml` is checked
6. Click **Finish**

#### Step 2: Wait for Maven Build
- Eclipse will download dependencies automatically
- Check progress in bottom-right corner
- Wait until "Building Workspace" completes

#### Step 3: Run
1. Open `src/main/java/com/example/OllamaClient.java`
2. Right-click in editor → **Run As → Java Application**
3. Output appears in **Console** view

#### Step 4: Create Run Configurations
1. **Run → Run Configurations**
2. Right-click **Java Application** → **New Configuration**
3. **Name**: `Ollama Client`
4. **Project**: `genai-clients`
5. **Main class**: Browse → `com.example.OllamaClient`
6. Click **Apply** and **Run**

Repeat for `LmStudioClient`.

### Troubleshooting Eclipse

#### Maven dependencies not resolving
- Right-click project → **Maven → Update Project**
- Check **Force Update of Snapshots/Releases**
- Click **OK**

#### Cannot find main class
- Right-click project → **Build Path → Configure Build Path**
- **Source** tab → Ensure `src/main/java` is listed
- Click **Apply**

---

## VS Code

### Prerequisites
Install these VS Code extensions:
1. **Extension Pack for Java** (Microsoft)
2. **Maven for Java** (Microsoft)
3. **Kotlin Language** (optional, for Kotlin files)

### Simple Examples

#### Step 1: Open Folder
1. **File → Open Folder**
2. Navigate to `day-1/JVM/simple-examples/`
3. Click **Select Folder**

#### Step 2: Open Java File
1. Click `GettingStartedOllama.java` in the explorer
2. VS Code will detect it's a Java file and activate Java extensions

#### Step 3: Run
1. Click the **Run** button (▶️) that appears above the `main()` method
2. Or press `F5`
3. Output appears in the **Terminal** panel

### Maven Project

#### Step 1: Open Maven Project
1. **File → Open Folder**
2. Navigate to `day-1/JVM/maven-example/`
3. Click **Select Folder**

#### Step 2: Wait for Java Extension
- VS Code will detect `pom.xml` and configure the project
- Wait for "Java projects: Ready" in the status bar
- Maven will download dependencies

#### Step 3: Run from Java Projects View
1. Click **Java Projects** icon in the left sidebar
2. Expand `genai-clients`
3. Right-click `com.example.OllamaClient`
4. Select **Run**

#### Step 4: Create launch.json (Optional)

For custom run configurations:

1. **Run → Add Configuration**
2. Select **Java**
3. VS Code creates `.vscode/launch.json`
4. Add configuration:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "java",
      "name": "Ollama Client",
      "request": "launch",
      "mainClass": "com.example.OllamaClient",
      "projectName": "genai-clients"
    },
    {
      "type": "java",
      "name": "LM Studio Client",
      "request": "launch",
      "mainClass": "com.example.LmStudioClient",
      "projectName": "genai-clients"
    }
  ]
}
```

5. Save the file
6. Select configuration from dropdown and press `F5`

### Troubleshooting VS Code

#### Java extension not working
- **Cmd+Shift+P** / **Ctrl+Shift+P** → "Java: Clean Java Language Server Workspace"
- Restart VS Code

#### Maven not detected
- Ensure `pom.xml` is at project root
- **Cmd+Shift+P** / **Ctrl+Shift+P** → "Maven: Update Project"

---

## NetBeans

### Simple Examples

#### Step 1: Create Java Project
1. **File → New Project**
2. **Categories**: Java with Ant
3. **Projects**: Java Application
4. Click **Next**
5. **Project Name**: `GenAI-Simple-Examples`
6. **Project Location**: Choose location
7. Uncheck **Create Main Class**
8. Click **Finish**

#### Step 2: Add Source File
1. Right-click **Source Packages**
2. **New → Java Class**
3. **Class Name**: `GettingStartedOllama`
4. Click **Finish**
5. Copy code from `simple-examples/GettingStartedOllama.java`
6. Paste into editor

#### Step 3: Run
1. Right-click the file → **Run File**
2. Or press `Shift+F6`
3. Output appears in **Output** panel

### Maven Project

#### Step 1: Open Maven Project
1. **File → Open Project**
2. Navigate to `day-1/JVM/maven-example/`
3. Select the folder (NetBeans will detect `pom.xml`)
4. Click **Open Project**

#### Step 2: Wait for Dependencies
- NetBeans will download Maven dependencies
- Check progress in bottom-right corner
- Wait for "Scanning projects" to complete

#### Step 3: Run
1. Expand **Source Packages → com.example**
2. Right-click `OllamaClient.java`
3. **Run File** (Shift+F6)
4. Output appears in **Output** panel

### Troubleshooting NetBeans

#### Maven dependencies not downloading
- Right-click project → **Build**
- Or: **Project Properties → Build → Dependencies** → Click **Reload**

---

## Common Setup for All IDEs

### 1. Ensure Servers are Running

Before running any example, start the appropriate server:

**For Ollama:**
```bash
ollama serve
ollama pull qwen3  # If not already downloaded
```

**For LM Studio:**
1. Open LM Studio application
2. Load a model (e.g., qwen3-4b-mlx)
3. Click the **Server** tab
4. Click **Start Server**

### 2. Verify Java Version

All IDEs need Java 11 or higher (Java 17+ recommended):

```bash
java -version
```

If you need to install Java:
- **macOS**: `brew install openjdk@17`
- **Windows**: Download from [Adoptium](https://adoptium.net/)
- **Linux**: `sudo apt install openjdk-17-jdk`

### 3. Configure Java in IDE

Make sure your IDE is configured to use the correct Java version:

- **IntelliJ**: File → Project Structure → Project → SDK
- **Eclipse**: Window → Preferences → Java → Installed JREs
- **VS Code**: Detected automatically, or set `java.home` in settings
- **NetBeans**: Tools → Java Platforms

---

## IDE Comparison

| Feature | IntelliJ IDEA | Eclipse | VS Code | NetBeans |
|---------|---------------|---------|---------|----------|
| **Maven Support** | Excellent | Good | Good | Excellent |
| **Kotlin Support** | Native | Plugin | Extension | Plugin |
| **Auto-import** | Yes | Yes | Yes | Yes |
| **Code Completion** | Excellent | Good | Good | Good |
| **Debugging** | Excellent | Excellent | Good | Excellent |
| **Learning Curve** | Medium | Medium | Low | Medium |
| **Performance** | Fast | Medium | Fast | Medium |
| **Free Version** | Community | Yes | Yes | Yes |

---

## Quick Reference Commands

### IntelliJ IDEA
```
Run:                  Ctrl+Shift+R (Mac) / Shift+F10 (Win)
Debug:                Ctrl+Shift+D (Mac) / Shift+F9 (Win)
Maven Reload:         Right-click pom.xml → Maven → Reload
```

### Eclipse
```
Run:                  Ctrl+F11
Debug:                F11
Maven Update:         Alt+F5
```

### VS Code
```
Run:                  F5
Debug:                F5
Open Command Palette: Cmd+Shift+P (Mac) / Ctrl+Shift+P (Win)
```

### NetBeans
```
Run File:             Shift+F6
Run Project:          F6
Debug:                Ctrl+F5
```

---

## Tips for All IDEs

### 1. Use Code Completion
Start typing and press:
- **IntelliJ/Eclipse/NetBeans**: `Ctrl+Space`
- **VS Code**: Automatically appears

### 2. View Documentation
Hover over any method/class to see Javadoc

### 3. Navigate to Definition
- **IntelliJ**: `Cmd+B` (Mac) / `Ctrl+B` (Win)
- **Eclipse**: `F3`
- **VS Code**: `F12`
- **NetBeans**: `Ctrl+Click`

### 4. Format Code
- **IntelliJ**: `Cmd+Alt+L` (Mac) / `Ctrl+Alt+L` (Win)
- **Eclipse**: `Ctrl+Shift+F`
- **VS Code**: `Shift+Alt+F`
- **NetBeans**: `Alt+Shift+F`

### 5. Organize Imports
- **IntelliJ**: `Ctrl+Alt+O`
- **Eclipse**: `Ctrl+Shift+O`
- **VS Code**: `Shift+Alt+O`
- **NetBeans**: `Ctrl+Shift+I`

---

## Next Steps

Once you can run the examples in your IDE:

1. **Experiment**: Modify prompts and models
2. **Debug**: Set breakpoints and step through code
3. **Extend**: Add new features like streaming or conversation history
4. **Build**: Create your own GenAI application

Happy coding! 🚀
