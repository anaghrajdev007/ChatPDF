import os
from flask import Flask, request, jsonify, send_from_directory
from pdf_ai import PDFConversationalAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(
    __name__, static_url_path="/static", static_folder=os.path.abspath("static")
)

pdf_ai = PDFConversationalAI()


@app.route("/manifest.json", methods=["GET", "POST"])
def manifest():

    return jsonify(
        {
            "name": "PDF Conversational AI",
            "description": "A conversational AI service that answers questions based on the content of uploaded PDFs.",
            "documentation": "",
            "endpoints": [
                {
                    "uri": "/",
                    "input_methods": ["GET"],
                    "input_query": "",
                    "input_headers": {},
                    "input_body": "",
                    "output": "",
                    "documentation": "",
                    "example_calls": [],
                    "is_public": True,
                },
                {
                    "uri": "/upload",
                    "description": "Upload PDFs for processing",
                    "input_methods": ["POST"],
                    "is_public": True,
                },
                {
                    "uri": "/question",
                    "description": "Ask a question based on the content of the uploaded PDFs",
                    "input_methods": ["POST"],
                    "is_public": True,
                },
            ],
            "is_public": True,
        }
    )


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    response, status_code = pdf_ai.process_pdfs(files)
    return jsonify(response), status_code


@app.route("/question", methods=["POST"])
def question():
    data = request.get_json()

    if request.headers.get("cost_only"):
        costs = {"min": 0, "max": 0, "estimated_cost": 5, "currency": "ProcessingUnits"}
        return jsonify({"cost": costs}), 200

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    response, status_code = pdf_ai.ask_question(data["question"])
    return jsonify(response), status_code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
