#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a server is running
check_server() {
    local url=$1
    local name=$2
    if curl -s --connect-timeout 2 "$url" >/dev/null 2>&1; then
        print_success "$name is running at $url"
        return 0
    else
        print_warning "$name is not running at $url"
        return 1
    fi
}

# Display help
show_help() {
    cat << EOF
Usage: ./run-examples.sh [OPTION]

Run GenAI JVM examples with different methods.

Options:
    simple-java-ollama      Run simple Java Ollama example
    simple-java-lmstudio    Run simple Java LM Studio example
    simple-kotlin-ollama    Run simple Kotlin Ollama example
    simple-kotlin-lmstudio  Run simple Kotlin LM Studio example
    maven-ollama            Run Maven Ollama example
    maven-lmstudio          Run Maven LM Studio example
    maven-build             Build Maven project
    maven-package           Package Maven project as JAR
    gradle-ollama           Run Gradle Ollama example
    gradle-lmstudio         Run Gradle LM Studio example
    gradle-build            Build Gradle project
    gradle-package          Package Gradle project as JAR
    check                   Check prerequisites
    help                    Show this help message

Examples:
    ./run-examples.sh simple-java-ollama
    ./run-examples.sh maven-ollama
    ./run-examples.sh gradle-ollama
    ./run-examples.sh check

EOF
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    echo ""
    
    local all_ok=true
    
    # Check Java
    if command_exists java; then
        local java_version=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        print_success "Java installed: $java_version"
    else
        print_error "Java not found. Please install Java 11 or higher."
        all_ok=false
    fi
    
    # Check Maven
    if command_exists mvn; then
        local maven_version=$(mvn -version 2>&1 | head -n 1)
        print_success "Maven installed: $maven_version"
    else
        print_warning "Maven not found. Maven examples will not work."
        echo "          Install with: brew install maven (macOS) or apt install maven (Linux)"
    fi
    
    # Check Gradle
    if [ -f "gradle-example/gradlew" ]; then
        print_success "Gradle wrapper found in gradle-example/"
    else
        print_warning "Gradle wrapper not found. Gradle examples may not work."
    fi
    
    # Check Kotlin
    if command_exists kotlinc; then
        local kotlin_version=$(kotlinc -version 2>&1 | head -n 1)
        print_success "Kotlin installed: $kotlin_version"
    else
        print_warning "Kotlin compiler not found. Kotlin examples will not work."
        echo "          Install with: brew install kotlin (macOS) or snap install kotlin --classic (Linux)"
    fi
    
    echo ""
    print_info "Checking servers..."
    echo ""
    
    # Check Ollama
    check_server "http://localhost:11434/api/tags" "Ollama"
    
    # Check LM Studio
    check_server "http://localhost:1234/v1/models" "LM Studio"
    
    echo ""
    if [ "$all_ok" = true ]; then
        print_success "All required tools are installed!"
    else
        print_error "Some required tools are missing. Please install them first."
    fi
}

# Run simple Java examples
run_simple_java() {
    local client=$1
    local file="simple-examples/GettingStarted${client}.java"
    
    if [ ! -f "$file" ]; then
        print_error "File not found: $file"
        exit 1
    fi
    
    print_info "Compiling $file..."
    javac "$file"
    
    if [ $? -eq 0 ]; then
        print_success "Compilation successful"
        print_info "Running GettingStarted${client}..."
        echo ""
        java -cp simple-examples "GettingStarted${client}"
        
        # Cleanup
        rm -f "simple-examples/GettingStarted${client}.class"
    else
        print_error "Compilation failed"
        exit 1
    fi
}

# Run simple Kotlin examples
run_simple_kotlin() {
    local client=$1
    local file="simple-examples/GettingStarted${client}.kt"
    local jar="${client,,}.jar"
    
    if [ ! -f "$file" ]; then
        print_error "File not found: $file"
        exit 1
    fi
    
    if ! command_exists kotlinc; then
        print_error "Kotlin compiler not found. Please install it first."
        exit 1
    fi
    
    print_info "Compiling $file..."
    kotlinc "$file" -include-runtime -d "$jar"
    
    if [ $? -eq 0 ]; then
        print_success "Compilation successful"
        print_info "Running $jar..."
        echo ""
        java -jar "$jar"
        
        # Cleanup
        rm -f "$jar"
    else
        print_error "Compilation failed"
        exit 1
    fi
}

# Run Maven examples
run_maven() {
    local profile=$1
    
    if ! command_exists mvn; then
        print_error "Maven not found. Please install Maven first."
        exit 1
    fi
    
    print_info "Running Maven example with profile: $profile"
    cd maven-example
    mvn exec:java -P"$profile"
    cd ..
}

# Build Maven project
build_maven() {
    if ! command_exists mvn; then
        print_error "Maven not found. Please install Maven first."
        exit 1
    fi
    
    print_info "Building Maven project..."
    cd maven-example
    mvn clean compile
    cd ..
    
    if [ $? -eq 0 ]; then
        print_success "Build successful"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Package Maven project
package_maven() {
    if ! command_exists mvn; then
        print_error "Maven not found. Please install Maven first."
        exit 1
    fi
    
    print_info "Packaging Maven project..."
    cd maven-example
    mvn clean package
    cd ..
    
    if [ $? -eq 0 ]; then
        print_success "Package created: maven-example/target/genai-clients-1.0.0.jar"
        print_info "Run with: java -jar maven-example/target/genai-clients-1.0.0.jar"
    else
        print_error "Packaging failed"
        exit 1
    fi
}

# Run Gradle examples
run_gradle() {
    local task=$1
    
    if [ ! -f "gradle-example/gradlew" ]; then
        print_error "Gradle wrapper not found. Please ensure gradle-example is set up correctly."
        exit 1
    fi
    
    print_info "Running Gradle example with task: $task"
    cd gradle-example
    ./gradlew "$task"
    cd ..
}

# Build Gradle project
build_gradle() {
    if [ ! -f "gradle-example/gradlew" ]; then
        print_error "Gradle wrapper not found. Please ensure gradle-example is set up correctly."
        exit 1
    fi
    
    print_info "Building Gradle project..."
    cd gradle-example
    ./gradlew build
    cd ..
    
    if [ $? -eq 0 ]; then
        print_success "Build successful"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Package Gradle project
package_gradle() {
    if [ ! -f "gradle-example/gradlew" ]; then
        print_error "Gradle wrapper not found. Please ensure gradle-example is set up correctly."
        exit 1
    fi
    
    print_info "Packaging Gradle project..."
    cd gradle-example
    ./gradlew clean jar
    cd ..
    
    if [ $? -eq 0 ]; then
        print_success "Package created: gradle-example/build/libs/gradle-example-1.0.0.jar"
        print_info "Run with: java -jar gradle-example/build/libs/gradle-example-1.0.0.jar"
    else
        print_error "Packaging failed"
        exit 1
    fi
}

# Main script
case "${1:-}" in
    simple-java-ollama)
        run_simple_java "Ollama"
        ;;
    simple-java-lmstudio)
        run_simple_java "LmStudio"
        ;;
    simple-kotlin-ollama)
        run_simple_kotlin "Ollama"
        ;;
    simple-kotlin-lmstudio)
        run_simple_kotlin "LmStudio"
        ;;
    maven-ollama)
        run_maven "ollama"
        ;;
    maven-lmstudio)
        run_maven "lmstudio"
        ;;
    maven-build)
        build_maven
        ;;
    maven-package)
        package_maven
        ;;
    gradle-ollama)
        run_gradle "runOllama"
        ;;
    gradle-lmstudio)
        run_gradle "runLmStudio"
        ;;
    gradle-build)
        build_gradle
        ;;
    gradle-package)
        package_gradle
        ;;
    check)
        check_prerequisites
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
