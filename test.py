from google import genai

client = genai.Client(api_key="AIzaSyCerjxXg_b6AAW4sQEq2Tzxo_sXV40dkOI")
response = client.models.generate_content(
    model="gemini-1.5-flash", contents="write IELTS band 5 essay on Many think that governments should fund programs in search of life on other planets. However, others believe governments should focus on unresolved issues on the planet. Provide your opinion and discuss both views."
)
print(response.text)