import requests

BASE_URL = "http://localhost:8000/api"

print("--- Testing Registration ---")
res = requests.post(f"{BASE_URL}/register", json={
    "name": "Test User",
    "email": "test@example.com",
    "password": "password123"
})
print(res.status_code, res.text)

print("--- Testing Login ---")
res = requests.post(f"{BASE_URL}/login", json={
    "email": "test@example.com",
    "password": "password123"
})
print(res.status_code, res.text)

print("--- Testing Answers ---")
res = requests.post(f"{BASE_URL}/answers", json={
    "email": "test@example.com",
    "answers": ["I love building web applications", "I have experience with Python and JavaScript", "I prefer backend over frontend"]
})
print(res.status_code, res.text)

# We can't really test the recommend endpoint fully without ollama running, so we'll skip it or just test if it returns a 500
# due to vector db being empty.
# print("--- Testing Recommend ---")
# res = requests.post(f"{BASE_URL}/recommend", json={
#     "email": "test@example.com",
#     "query": "Suggest a project"
# })
# print(res.status_code, res.text)

