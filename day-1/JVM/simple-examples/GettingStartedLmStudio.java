import java.io.*;
import java.net.*;

public class GettingStartedLmStudio {
    public static void main(String[] args) {
        try {
            String jsonString = "{\"model\":\"qwen3-4b-mlx\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}";

            URI uri = URI.create("http://localhost:1234/v1/chat/completions");
            HttpURLConnection connection = (HttpURLConnection) uri.toURL().openConnection();

            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(15000);

            try (OutputStreamWriter writer = new OutputStreamWriter(connection.getOutputStream())) {
                writer.write(jsonString);
                writer.flush();
            }

            int statusCode = connection.getResponseCode();

            String responseBody;
            if (statusCode >= 200 && statusCode < 300) {
                try (InputStream inputStream = connection.getInputStream()) {
                    ByteArrayOutputStream result = new ByteArrayOutputStream();
                    byte[] buffer = new byte[1024];
                    int length;
                    while ((length = inputStream.read(buffer)) != -1) {
                        result.write(buffer, 0, length);
                    }
                    responseBody = result.toString("UTF-8");
                }
            } else {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getErrorStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }
                    responseBody = response.toString();
                }
                System.out.println("Error response: " + responseBody);
                return;
            }

            String content = extractMessageContent(responseBody);
            System.out.println(content.strip());

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static String extractMessageContent(String json) {
        // Look for the content field more flexibly
        int contentStart = json.indexOf("\"content\":");
        if (contentStart == -1) return "";

        // Find the opening quote of the content value
        int quoteStart = json.indexOf("\"", contentStart + "\"content\":".length());
        if (quoteStart == -1) return "";

        int valueStart = quoteStart + 1;
        int contentEnd = valueStart;

        // Find the end of the content field, accounting for escaped quotes
        while (contentEnd < json.length()) {
            char c = json.charAt(contentEnd);

            if (c == '\\') {
                // Skip escaped character
                contentEnd += 2;
                continue;
            }

            if (c == '"') {
                // Found the end of the content string
                break;
            }
            contentEnd++;
        }

        if (contentEnd >= json.length()) return "";

        String content = json.substring(valueStart, contentEnd);
        return content.replace("\\n", "\n").replace("\\\"", "\"").replace("\\t", "\t");
    }
}