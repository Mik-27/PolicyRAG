from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit

from application import PolicyRAG

app = Flask(__name__)
socketio = SocketIO(app)

rag = PolicyRAG()

@app.route('/chat')
def chat():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(data):
    user_message = data['message']
    print(user_message)
    
    # Process the user message
    # response = generate_chatbot_response(user_message)
    response = "Test"
    
    # Send the response back to the user
    emit('response', {'message': response})

def generate_chatbot_response(message):
    # A simple logic to demonstrate (replace with more advanced NLP if needed)
    if "search" in message.lower():
        # Simulate a search or perform one based on extracted keywords
        query_embedding = rag.generate_embedding(message)
        results = rag.search_docs(by="embedding", query=query_embedding)
        return f"Found {len(results)} results for your search."
    else:
        return "I'm here to help! You can ask me to search for PDFs."

if __name__ == '__main__':
    socketio.run(app, debug=True)
