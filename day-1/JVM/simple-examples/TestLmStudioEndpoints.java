import java.io.*;
import java.net.*;

public class TestLmStudioEndpoints {
    public static void main(String[] args) {
        System.out.println("=== Testing LM Studio Chat Completions ===\n");
        
        // Test with the correct model name from the diagnostic
        String[] endpoints = {
            "http://localhost:1234/v1/chat/completions",
            "http://localhost:1234/chat/completions",
            "http://localhost:1234/v1/completions"
        };
        
        String[] models = {
            "qwen/qwen3-coder-30b",
            "qwen3-coder-30b",
            "qwen3-4b-mlx"
        };
        
        for (String endpoint : endpoints) {
            System.out.println("Testing endpoint: " + endpoint);
            for (String model : models) {
                System.out.println("  With model: " + model);
                testChatCompletion(endpoint, model);
            }
            System.out.println();
        }
    }
    
    private static void testChatCompletion(String endpoint, String model) {
        try {
            String jsonString = "{\"model\":\"" + model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":50}";
            
            URL url = new URL(endpoint);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            connection.setConnectTimeout(3000);
            connection.setReadTimeout(10000);
            
            try (OutputStreamWriter writer = new OutputStreamWriter(connection.getOutputStream())) {
                writer.write(jsonString);
                writer.flush();
            }
            
            int statusCode = connection.getResponseCode();
            
            if (statusCode >= 200 && statusCode < 300) {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getInputStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    System.out.println("    ✓ SUCCESS! Status: " + statusCode);
                    System.out.println("    Response: " + response.toString().substring(0, Math.min(200, response.length())) + "...");
                }
            } else {
                System.out.println("    ❌ FAILED! Status: " + statusCode);
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(connection.getErrorStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    System.out.println("    Error: " + response.toString());
                }
            }
        } catch (Exception e) {
            System.out.println("    ❌ Exception: " + e.getMessage());
        }
    }
}
