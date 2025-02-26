import os
from flask import Flask, request, jsonify
from pdf_ai import PDFConversationalAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
pdf_ai = PDFConversationalAI()

@app.route('/')
def manifest():
    return jsonify({
        "name": "PDF Conversational AI",
        "description": "A conversational AI service that answers questions based on the content of uploaded PDFs.",
        "documentation": """
            <h3>Endpoints</h3>
            <ul>
                <li>POST /upload: Upload PDFs to process the text content.</li>
                <li>POST /question: Ask a question based on the uploaded PDFs.</li>
            </ul>
        """
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    response, status_code = pdf_ai.process_pdfs(files)
    return jsonify(response), status_code

@app.route('/question', methods=['POST'])
def question():
    data = request.get_json()
    
    if request.headers.get('cost_only') :
        costs = {"min": 0, "max": 0, "estimated_cost": 5, "currency": "ProcessingUnits"}
        return jsonify({"cost": costs}), 200
    
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    response, status_code = pdf_ai.ask_question(data['question'])
    return jsonify(response), status_code

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
