plugins {
    java
    application
}

group = "com.example"
version = "1.0.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.10.1")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

// Configure source sets to include the simple-examples directory
sourceSets {
    main {
        java {
            srcDirs("src/main/java", "../simple-examples")
        }
    }
}

// Default application class
application {
    mainClass.set("com.example.OllamaClient")
}

// Create custom tasks for running each client
tasks.register<JavaExec>("runOllama") {
    group = "application"
    description = "Run the Ollama client"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.OllamaClient")
}

tasks.register<JavaExec>("runLmStudio") {
    group = "application"
    description = "Run the LM Studio client"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("com.example.LmStudioClient")
}

// Configure the default run task to use OllamaClient
tasks.named<JavaExec>("run") {
    mainClass.set("com.example.OllamaClient")
}

// Configure JAR task to create executable JAR
tasks.jar {
    manifest {
        attributes["Main-Class"] = "com.example.OllamaClient"
    }
    // Include dependencies in the JAR
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}
