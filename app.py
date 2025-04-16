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
    # print(user_message, type(user_message))
    
    # Process the user message
    response = generate_chatbot_response(user_message)
    # print(response)
    
    # Send the response back to the user
    emit('response', {'message': response})

def generate_chatbot_response(message):
    docs = rag.search_docs(by="embedding", query=message)
    text = docs[0]['text']
    res = rag.generate_query_output(query=message, context=text)
    
    return res

if __name__ == '__main__':
    socketio.run(app, debug=True)
