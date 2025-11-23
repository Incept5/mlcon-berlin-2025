import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class GettingStartedOllama {
    public static void main(String[] args) {
        try {
            HttpClient client = HttpClient.newHttpClient();

            String jsonString = "{\"model\":\"qwen3\",\"prompt\":\"Hello\",\"stream\":false,\"Think\":false}";

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:11434/api/generate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonString))
                .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            String responseBody = response.body();
            String responseText = extractJsonValue(responseBody, "response");
            System.out.println(responseText);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static String extractJsonValue(String json, String key) {
        String searchKey = "\"" + key + "\":\"";
        int startIndex = json.indexOf(searchKey);
        if (startIndex == -1) return "";

        startIndex += searchKey.length();
        int endIndex = json.indexOf("\"", startIndex);
        if (endIndex == -1) return "";

        return json.substring(startIndex, endIndex).replace("\\n", "\n").replace("\\\"", "\"");
    }
}