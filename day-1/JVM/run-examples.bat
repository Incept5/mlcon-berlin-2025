@echo off
REM Windows batch script to run JVM examples

setlocal EnableDelayedExpansion

REM Color codes don't work well in batch, so we'll use simple prefixes
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

if "%1"=="" goto show_help
if "%1"=="help" goto show_help
if "%1"=="--help" goto show_help
if "%1"=="-h" goto show_help

REM Check prerequisites
if "%1"=="check" goto check_prerequisites

REM Simple Java examples
if "%1"=="simple-java-ollama" goto simple_java_ollama
if "%1"=="simple-java-lmstudio" goto simple_java_lmstudio

REM Maven examples
if "%1"=="maven-ollama" goto maven_ollama
if "%1"=="maven-lmstudio" goto maven_lmstudio
if "%1"=="maven-build" goto maven_build
if "%1"=="maven-package" goto maven_package

echo %ERROR% Unknown option: %1
echo.
goto show_help

:show_help
echo Usage: run-examples.bat [OPTION]
echo.
echo Run GenAI JVM examples with different methods.
echo.
echo Options:
echo     simple-java-ollama      Run simple Java Ollama example
echo     simple-java-lmstudio    Run simple Java LM Studio example
echo     maven-ollama            Run Maven Ollama example
echo     maven-lmstudio          Run Maven LM Studio example
echo     maven-build             Build Maven project
echo     maven-package           Package Maven project as JAR
echo     check                   Check prerequisites
echo     help                    Show this help message
echo.
echo Examples:
echo     run-examples.bat simple-java-ollama
echo     run-examples.bat maven-ollama
echo     run-examples.bat check
echo.
goto end

:check_prerequisites
echo %INFO% Checking prerequisites...
echo.

where java >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo %SUCCESS% Java is installed
    java -version
) else (
    echo %ERROR% Java not found. Please install Java 11 or higher.
)

echo.
where mvn >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo %SUCCESS% Maven is installed
    mvn -version 2>&1 | findstr "Apache Maven"
) else (
    echo %WARNING% Maven not found. Maven examples will not work.
    echo           Install with: choco install maven
)

echo.
echo %INFO% To check if servers are running:
echo     Ollama: curl http://localhost:11434/api/tags
echo     LM Studio: curl http://localhost:1234/v1/models
echo.
goto end

:simple_java_ollama
echo %INFO% Compiling GettingStartedOllama.java...
cd simple-examples
javac GettingStartedOllama.java
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% Compilation failed
    cd ..
    exit /b 1
)
echo %SUCCESS% Compilation successful
echo %INFO% Running GettingStartedOllama...
echo.
java GettingStartedOllama
del GettingStartedOllama.class
cd ..
goto end

:simple_java_lmstudio
echo %INFO% Compiling GettingStartedLmStudio.java...
cd simple-examples
javac GettingStartedLmStudio.java
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% Compilation failed
    cd ..
    exit /b 1
)
echo %SUCCESS% Compilation successful
echo %INFO% Running GettingStartedLmStudio...
echo.
java GettingStartedLmStudio
del GettingStartedLmStudio.class
cd ..
goto end

:maven_ollama
where mvn >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% Maven not found. Please install Maven first.
    exit /b 1
)
echo %INFO% Running Maven example with profile: ollama
cd maven-example
mvn exec:java -Pollama
cd ..
goto end

:maven_lmstudio
where mvn >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% Maven not found. Please install Maven first.
    exit /b 1
)
echo %INFO% Running Maven example with profile: lmstudio
cd maven-example
mvn exec:java -Plmstudio
cd ..
goto end

:maven_build
where mvn >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% Maven not found. Please install Maven first.
    exit /b 1
)
echo %INFO% Building Maven project...
cd maven-example
mvn clean compile
if %ERRORLEVEL% EQU 0 (
    echo %SUCCESS% Build successful
) else (
    echo %ERROR% Build failed
)
cd ..
goto end

:maven_package
where mvn >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% Maven not found. Please install Maven first.
    exit /b 1
)
echo %INFO% Packaging Maven project...
cd maven-example
mvn clean package
if %ERRORLEVEL% EQU 0 (
    echo %SUCCESS% Package created: maven-example\target\genai-clients-1.0.0.jar
    echo %INFO% Run with: java -jar maven-example\target\genai-clients-1.0.0.jar
) else (
    echo %ERROR% Packaging failed
)
cd ..
goto end

:end
endlocal
