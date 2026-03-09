# Graduation Projects Generator API Documentation

This document provides all the necessary details to build a Flutter application that interacts with the "Graduation Projects Generator" AI Backend. The backend is built using FastAPI, LangChain, and Ollama.

## Base Information
- **Base URL:** `http://<YOUR_SERVER_IP>:8000` *(Replace `<YOUR_SERVER_IP>` with the IP address of the machine running the backend. For local testing on an Android emulator, use `10.0.2.2`. For iOS simulator, use `127.0.0.1` or `localhost`.)*
- **Content-Type:** `application/json`

---

## Endpoints

### 1. Generate Project Idea / Ask Question
This is the primary (and currently only) endpoint used to interact with the AI assistant. It allows the user to ask for project ideas or ask specific technical questions, and returns an AI-generated response based on the dataset of projects.

- **Endpoint:** `/api/chat`
- **Method:** `POST`
- **Description:** Sends a user query to the AI and receives a formatted response in Arabic.

#### Request Body (JSON)

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | The user's input, question, or request (e.g., "أريد مشروع تخرج في مجال الذكاء الاصطناعي الطبي"). |
| `conversation_history` | `string` | Optional | The previous conversation context (if implemented in the frontend) to help the AI remember the context of the chat. Default is an empty string `""`. |

**Example Request Payload:**
```json
{
  "query": "اقترح مشروع تخرج يستخدم تقنيات التعلم العميق",
  "conversation_history": "الطالب: ما هي التقنيات الحديثة؟\nالمساعد: تقنيات التعلم العميق..."
}
```

#### Response Body (JSON)

The response will be a JSON object containing the AI's answer.

| Field | Type | Description |
| :--- | :--- | :--- |
| `message` | `string` | The AI-generated text response. This text will often contain markdown formatting (like bolding, lists, etc.) so the Flutter app should render it using a package like `flutter_markdown`. |

**Example Response Payload (Success - 200 OK):**
```json
{
  "message": "عنوان المشروع: التشخيص المبكر للأمراض الجلدية باستخدام التعلم العميق...\n\nوصف المشروع: ..."
}
```

#### Error Responses

- **500 Internal Server Error:** Returned if the vector database failed to initialize properly on the backend.
  ```json
  {
    "detail": "قاعدة البيانات غير مهيأة بعد."
  }
  ```

---

## Flutter Implementation Guidelines (For the AI Developer)

When building the Flutter application, please adhere to the following guidelines:

1. **State Management:** Use a robust state management solution (like Provider, Riverpod, or Bloc) to handle the chat message list and loading states.
2. **HTTP Client:** Use the `http` or `dio` package to make requests to the `/api/chat` endpoint.
3. **Markdown Rendering:** The `message` returned by the API is typically formatted using Markdown. Use the `flutter_markdown` package to beautifully render the text in the chat bubbles.
4. **Chat Interface Model:** Create a simple `ChatMessage` model to distinguish between "User" and "AI" messages.
   ```dart
   class ChatMessage {
     final String text;
     final bool isUser;
     ChatMessage({required this.text, required this.isUser});
   }
   ```
5. **Loading Indicators:** Show a clear loading indicator (e.g., `CircularProgressIndicator` or a typing indicator animation) while waiting for the POST request to complete, as AI generation can take a few moments.
6. **Error Handling:** Implement `try-catch` blocks around the API call. If a connection error occurs or a 500 status code is returned, display a user-friendly Snackbar or dialog (e.g., "تعذر الاتصال بالخادم الذكي").
7. **CORS:** The backend already has CORS configured to accept all origins, so you will not face CORS issues during development or deployment.
8. **RTL Support:** Ensure the UI directionality is set to `TextDirection.rtl` since the target language of the AI is Arabic.

## Example Flutter API Call

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Replace with your actual backend IP
  static const String baseUrl = 'http://10.0.2.2:8000'; 

  static Future<String?> sendMessage(String query, {String history = ""}) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/chat'),
        headers: {'Content-Type': 'application/json; charset=UTF-8'},
        body: jsonEncode({
          'query': query,
          'conversation_history': history,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        return data['message'];
      } else {
        throw Exception('Failed to load response: ${response.statusCode}');
      }
    } catch (e) {
      print('Error: $e');
      return "عذراً، حدث خطأ في الاتصال بالخادم.";
    }
  }
}
```
