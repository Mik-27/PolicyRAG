from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import time

from application import PolicyRAG

app = Flask(__name__)
socketio = SocketIO(app)

rag = PolicyRAG()

@app.route('/chat')
def chat():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(data):
    start = time.time()
    user_message = data['message']
    # print(user_message, type(user_message))
    
    # Process the user message
    response = generate_chatbot_response(user_message)
    end = time.time()
    print("Time taken:", end-start)
    # print(response)
    
    # Send the response back to the user
    emit('response', {'message': response})

def generate_chatbot_response(message):
    docs = rag.search_docs(by="embedding", query=message)
    relevant_docs = [doc for doc in docs if doc['score'] > 0.7]
    print("Relevant documents:", len(relevant_docs))
    
    text = docs[0]['text']
    res = rag.generate_query_output(query=message, context=text)
    if relevant_docs:
        combined_text = ""
        for i, doc in enumerate(relevant_docs):
            combined_text += f"\n--- Document {i+1} ---\n{doc['text']}\n"
            
        res = rag.generate_query_output(query=message, context=combined_text)
    else:
        res = "I couldn't find any relevant information about that topic in the policies."
    
    return res

if __name__ == '__main__':
    socketio.run(app, debug=True)
