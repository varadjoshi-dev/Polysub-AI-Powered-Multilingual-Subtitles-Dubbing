import os
import uuid
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import the process function and safe_filename utility from your subtitle module
from subtitle import process, safe_filename

# --- Global In-Memory Job Store ---
# The jobs dictionary now lives here, in the main server file.
jobs = {}

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No valid file part")

    job_id = str(uuid.uuid4())
    safe_original_filename = safe_filename(file.filename)
    filename = f"{job_id}_{safe_original_filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    file_size = os.path.getsize(filepath)

    jobs[job_id] = {
        "id": job_id,
        "size": file_size,
        "filename": file.filename,
        "server_path": filepath,
        "status": "uploaded",
        "progress": 0,
        "logs": ["File uploaded successfully."],
    }

    return JSONResponse(content={
        "filename": file.filename,
        "size": file_size,
        "path": filepath,
        "id": job_id
    })

@app.post("/api/process")
async def start_process(payload: dict, background_tasks: BackgroundTasks):
    job_id = payload.get("jobId")
    file_path = payload.get("filePath")
    langs = payload.get("langs", ["hin_Deva"])
    enable_tts = payload.get("enableTts", False)

    if not job_id or not file_path:
        raise HTTPException(status_code=400, detail="Missing jobId or filePath")

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    jobs[job_id]["status"] = "queued"
    jobs[job_id]["logs"].append("Job queued for processing.")
    jobs[job_id]["opts"] = {"enableTts": enable_tts, "langs": langs}

    # *** FIX: Pass the 'jobs' dictionary to the process function ***
    background_tasks.add_task(
        process,
        jobs=jobs, # Pass the jobs dictionary
        input_path=file_path,
        job_id=job_id,
        enableTts=enable_tts,
        target_lang=",".join(langs)
    )

    return JSONResponse(content=jobs[job_id])


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") == "done" and not job.get("output_files"):
        server_path = job.get("server_path", "")
        base_name = os.path.splitext(os.path.basename(server_path))[0]
        
        output_files = [
            f for f in os.listdir('.')
            if f.startswith(base_name) and (f.endswith('.mp4') or f.endswith('.srt'))
        ]
        job["output_files"] = output_files

    return JSONResponse(content=job)

@app.get("/api/download/{filename}")
def download_file(filename: str):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(path=filename, media_type='application/octet-stream', filename=filename)

# To run: uvicorn routes:app --reload

