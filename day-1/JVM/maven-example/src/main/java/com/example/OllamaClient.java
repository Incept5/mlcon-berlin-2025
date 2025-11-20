package com.example;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;

public class OllamaClient {

    private static final String OLLAMA_URL = "http://localhost:11434/api/generate";
    private static final MediaType JSON = MediaType.get("application/json");

    public static void main(String[] args) {
        OkHttpClient client = new OkHttpClient();

        JsonObject requestJson = new JsonObject();
        requestJson.addProperty("model", "qwen3");
        requestJson.addProperty("prompt", "Hello");
        requestJson.addProperty("stream", false);
        requestJson.addProperty("Think", false);

        RequestBody body = RequestBody.create(requestJson.toString(), JSON);
        Request request = new Request.Builder()
                .url(OLLAMA_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                String responseBody = response.body().string();
                JsonObject jsonResponse = JsonParser.parseString(responseBody).getAsJsonObject();
                String content = jsonResponse.get("response").getAsString();
                System.out.println(content);
            } else {
                System.err.println("Request failed: " + response.code());
            }
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        } finally {
            // Shutdown the client to clean up threads
            client.dispatcher().executorService().shutdown();
            client.connectionPool().evictAll();
            try {
                // Wait a moment for threads to finish
                if (!client.dispatcher().executorService().awaitTermination(1, java.util.concurrent.TimeUnit.SECONDS)) {
                    client.dispatcher().executorService().shutdownNow();
                }
            } catch (InterruptedException e) {
                client.dispatcher().executorService().shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}