from flask import Flask, request, jsonify, send_from_directory
import os, time
from pathlib import Path
from pushup_counter import run
import pandas as pd

app = Flask(__name__)

# folders
UPLOAD_FOLDER, RESULTS_FOLDER = Path("uploads"), Path("results")
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # save upload
    filepath = UPLOAD_FOLDER / file.filename
    file.save(str(filepath))

    # unique run folder
    run_id = f"run_{int(time.time())}"
    out_dir = RESULTS_FOLDER / run_id
    out_dir.mkdir()

    # run pushup counter
    run(str(filepath), use_webcam=False, out_dir=out_dir, frame_csv=True, no_window=True)

    # outputs
    video_file, rep_file = out_dir / "annotated.mp4", out_dir / "per_rep.csv"

    # build report
    report = pd.read_csv(rep_file).to_dict(orient="records") if rep_file.exists() else []

    return jsonify({
        "message": "done",
        "annotated_video": f"/results/{run_id}/annotated.mp4",
        "rep_report": report
    })

# serve results (videos, csv, etc.)
@app.route("/results/<run_id>/<filename>")
def results(run_id, filename):
    directory = (RESULTS_FOLDER / run_id).resolve()
    return send_from_directory(directory, filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
