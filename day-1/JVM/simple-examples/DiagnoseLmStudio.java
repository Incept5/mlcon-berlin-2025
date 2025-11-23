import java.io.*;
import java.net.*;

public class DiagnoseLmStudio {
    public static void main(String[] args) {
        System.out.println("=== LM Studio Diagnostic Tool ===\n");
        
        // Test 1: Check if server is running
        System.out.println("Test 1: Checking if LM Studio server is running...");
        boolean serverRunning = testConnection("http://localhost:1234");
        if (!serverRunning) {
            System.out.println("❌ FAILED: Cannot connect to http://localhost:1234");
            System.out.println("\nPossible solutions:");
            System.out.println("1. Open LM Studio application");
            System.out.println("2. Go to 'Local Server' or 'Server' tab");
            System.out.println("3. Click 'Start Server' button");
            System.out.println("4. Verify server is running on port 1234");
            return;
        }
        System.out.println("✓ Server is running on localhost:1234\n");
        
        // Test 2: Check available models endpoint
        System.out.println("Test 2: Checking /v1/models endpoint...");
        String modelsResponse = makeGetRequest("http://localhost:1234/v1/models");
        if (modelsResponse != null) {
            System.out.println("✓ /v1/models endpoint works!");
            System.out.println("Response: " + modelsResponse + "\n");
        } else {
            System.out.println("❌ /v1/models endpoint failed\n");
        }
        
        // Test 3: Try alternative models endpoint
        System.out.println("Test 3: Checking /models endpoint...");
        String altModelsResponse = makeGetRequest("http://localhost:1234/models");
        if (altModelsResponse != null) {
            System.out.println("✓ /models endpoint works!");
            System.out.println("Response: " + altModelsResponse + "\n");
        } else {
            System.out.println("❌ /models endpoint not available\n");
        }
        
        // Test 4: Try the chat completions endpoint with the expected model
        System.out.println("Test 4: Testing /v1/chat/completions endpoint...");
        testChatCompletions("http://localhost:1234/v1/chat/completions", "qwen3-4b-mlx");
        
        // Test 5: Try alternative endpoint paths
        System.out.println("\nTest 5: Testing alternative endpoint paths...");
        testChatCompletions("http://localhost:1234/chat/completions", "qwen3-4b-mlx");
        testChatCompletions("http://localhost:1234/completions", "qwen3-4b-mlx");
        
        System.out.println("\n=== Diagnostic Complete ===");
        System.out.println("\nIf all tests failed, please check:");
        System.out.println("1. LM Studio is actually running");
        System.out.println("2. A model is loaded in LM Studio");
        System.out.println("3. The server port in LM Studio settings (should be 1234)");
        System.out.println("4. Try accessing http://localhost:1234 in your browser");
    }
    
    private static boolean testConnection(String urlString) {
        try {
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(2000);
            connection.setReadTimeout(2000);
            int responseCode = connection.getResponseCode();
            return responseCode > 0;
        } catch (Exception e) {
            return false;
        }
    }
    
    private static String makeGetRequest(String urlString) {
        try {
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(3000);
            connection.setReadTimeout(3000);
            
            int statusCode = connection.getResponseCode();
            
            if (statusCode >= 200 && statusCode < 300) {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getInputStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    return response.toString();
                }
            } else {
                System.out.println("  Status code: " + statusCode);
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getErrorStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    System.out.println("  Error: " + response.toString());
                }
                return null;
            }
        } catch (Exception e) {
            System.out.println("  Exception: " + e.getMessage());
            return null;
        }
    }
    
    private static void testChatCompletions(String endpoint, String model) {
        try {
            String jsonString = "{\"model\":\"" + model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
            
            URL url = new URL(endpoint);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            connection.setConnectTimeout(3000);
            connection.setReadTimeout(5000);
            
            try (OutputStreamWriter writer = new OutputStreamWriter(connection.getOutputStream())) {
                writer.write(jsonString);
                writer.flush();
            }
            
            int statusCode = connection.getResponseCode();
            System.out.println("  Endpoint: " + endpoint);
            System.out.println("  Status code: " + statusCode);
            
            if (statusCode >= 200 && statusCode < 300) {
                System.out.println("  ✓ SUCCESS! This endpoint works.");
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getInputStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    int lines = 0;
                    while ((line = reader.readLine()) != null && lines++ < 5) {
                        response.append(line).append("\n");
                    }
                    System.out.println("  Response preview: " + response.toString());
                }
            } else if (statusCode == 404) {
                System.out.println("  ❌ 404 Not Found - This endpoint doesn't exist");
            } else {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getErrorStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    System.out.println("  Error response: " + response.toString());
                }
            }
            System.out.println();
        } catch (Exception e) {
            System.out.println("  Exception: " + e.getMessage());
            System.out.println();
        }
    }
}
