from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import tempfile
import threading
import time
from jobs import jobs
from subtitle import process

app = Flask(__name__)
CORS(app)  # Enable CORS for the frontend to interact with the backend

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# jobs = {}  # { job_id: {"filename": str, "status": str, "output": str} }
uploaded = {} # { filename: }

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    job_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Get file size in bytes
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size} bytes")

    # Create a job
    job_id = str(uuid.uuid4())

  # initialize job in global dict
    jobs[job_id] = {
        "id": job_id,
        "size": file_size,
        "filename": file.filename,
        "status": "queued",
        "progress": 0,
        "logs": [],
    }

    return jsonify({
        "filename": file.filename,
        "size": os.path.getsize(filepath),
        "path": filepath,
        "id": job_id
    })

@app.route("/api/process", methods=["POST"])
def start_process():

    payload = request.get_json()
    if not payload:
       return jsonify({"error": "No JSON payload received"}), 400

    file_path = payload.get("filePath")
    job_id = payload.get("jobId")
    enableTts = payload.get("enableTts")
    enableRealtime = payload.get("enableRealtime", False)
    generateSrt = payload.get("generateSrt", True)
    langs = payload.get("langs", [])

    # Run the pipeline in background
    threading.Thread(
              target=process,
              args=(file_path,),  # positional args: input_path
              kwargs={
                  "job_id": job_id,
                  "enableTts": enableTts,
                  "enableRealtime": enableRealtime,
                  "generateSrt": generateSrt,
                  "target_lang": langs
              },
              daemon=True
          ).start()

    jobs[job_id].update({"opts": { "generateSrt": generateSrt, "enableTts": enableTts, "enableRealtime": enableRealtime },
                    "langs": langs })

    return jobs[job_id]


# --- Check job status ---
@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
