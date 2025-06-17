from pyngrok import ngrok

public_url = ngrok.connect(3000)
print(f"🌍 Frontend accessibile su: {public_url}")
