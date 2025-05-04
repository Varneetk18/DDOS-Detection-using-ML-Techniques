from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import numpy as np
import pickle
import os
import tempfile
from pathlib import Path
import time
from fastapi import HTTPException
from scapy.all import sniff, IP, TCP, UDP, ICMP
from threading import Thread, Lock
from collections import defaultdict
import json
from scapy.arch import get_if_list
import psutil
from scapy.arch.windows import get_windows_if_list
import socket

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # Create uploads directory if it doesn't exist

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global variables for packet capture
PACKET_BUFFER = defaultdict(lambda: {
    'packets': 0,
    'bytes': 0,
    'start_time': None,
    'protocols': defaultdict(int),
    'dest_ports': defaultdict(int),
    'src_ports': defaultdict(int),
    'unique_ips': set()
})
PACKET_LOCK = Lock()
IS_CAPTURING = False
CAPTURE_THREAD = None

# Load models once at startup
def load_models():
    models = {}
    try:
        with open("support.pkl", "rb") as f:
            models['svm'] = pickle.load(f)
        with open("dt.pkl", "rb") as f:
            models['dt'] = pickle.load(f)
        with open("nb.pkl", "rb") as f:
            models['nb'] = pickle.load(f)
        with open("rf.pkl", "rb") as f:
            models['rf'] = pickle.load(f)
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

MODELS = load_models()
PREDICTION_MAPPING = {0: 'Normal', 1: 'DDoS Attack', 2: 'Probe Attack'}

def retrain_and_save_models():
    """Retrain and save models with current scikit-learn version"""
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import pickle

    try:
        
        data = pd.read_csv("training_data.csv")
        X = data.drop('label', axis=1) 
        y = data['label']

        # Train models
        models = {
            'svm': LinearSVC(random_state=42),
            'dt': DecisionTreeClassifier(random_state=42),
            'nb': GaussianNB(),
            'rf': RandomForestClassifier(random_state=42)
        }

        for name, model in models.items():
            model.fit(X, y)
            with open(f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
        
        print("Models retrained and saved successfully!")
        return True
    except Exception as e:
        print(f"Error retraining models: {e}")
        return False

# Add this right after your imports
if not load_models():
    print("Attempting to retrain models...")
    retrain_and_save_models()
    MODELS = load_models()

def prepare_features(data, is_manual=True):
    """Prepare features for prediction"""
    try:
        if is_manual:
            # For manual input, create DataFrame with all required features
            input_data = pd.DataFrame([{
                'duration': float(data.get('duration', 0)),
                'protocol_type': data.get('protocol_type', 'tcp'),
                'service': data.get('service', 'http'),
                'flag': data.get('flag', 'SF'),
                'src_bytes': float(data.get('src_bytes', 0)),
                'dst_bytes': float(data.get('dst_bytes', 0)),
                'land': int(data.get('land', 0)),
                'wrong_fragment': float(data.get('wrong_fragment', 0)),
                'urgent': float(data.get('urgent', 0)),
                'hot': float(data.get('hot', 0)),
                'num_failed_logins': float(data.get('num_failed_logins', 0)),
                'logged_in': int(data.get('logged_in', 0)),
                'num_compromised': float(data.get('num_compromised', 0)),
                'root_shell': float(data.get('root_shell', 0)),
                'su_attempted': float(data.get('su_attempted', 0)),
                'num_root': float(data.get('num_root', 0)),
                'num_file_creations': float(data.get('num_file_creations', 0)),
                'num_shells': float(data.get('num_shells', 0)),
                'num_access_files': float(data.get('num_access_files', 0)),
                'num_outbound_cmds': float(data.get('num_outbound_cmds', 0)),
                'is_host_login': int(data.get('is_host_login', 0)),
                'is_guest_login': int(data.get('is_guest_login', 0)),
                'count': float(data.get('count', 0)),
                'srv_count': float(data.get('srv_count', 0)),
                'serror_rate': float(data.get('serror_rate', 0)),
                'srv_serror_rate': float(data.get('srv_serror_rate', 0)),
                'rerror_rate': float(data.get('rerror_rate', 0)),
                'srv_rerror_rate': float(data.get('srv_rerror_rate', 0)),
                'same_srv_rate': float(data.get('same_srv_rate', 0)),
                'diff_srv_rate': float(data.get('diff_srv_rate', 0)),
                'srv_diff_host_rate': float(data.get('srv_diff_host_rate', 0)),
                'dst_host_count': float(data.get('dst_host_count', 0)),
                'dst_host_srv_count': float(data.get('dst_host_srv_count', 0)),
                'dst_host_same_srv_rate': float(data.get('dst_host_same_srv_rate', 0)),
                'dst_host_diff_srv_rate': float(data.get('dst_host_diff_srv_rate', 0)),
                'dst_host_same_src_port_rate': float(data.get('dst_host_same_src_port_rate', 0)),
                'dst_host_srv_diff_host_rate': float(data.get('dst_host_srv_diff_host_rate', 0)),
                'dst_host_serror_rate': float(data.get('dst_host_serror_rate', 0)),
                'dst_host_srv_serror_rate': float(data.get('dst_host_srv_serror_rate', 0)),
                'dst_host_rerror_rate': float(data.get('dst_host_rerror_rate', 0)),
                'dst_host_srv_rerror_rate': float(data.get('dst_host_srv_rerror_rate', 0))
            }])
        else:
            # For file upload, use the provided DataFrame
            input_data = data.copy()
            
        # Get categorical columns
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Create dummy variables
        input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols)
        
        # Ensure all required columns exist
        if hasattr(MODELS['svm'], 'feature_names_in_'):
            required_features = MODELS['svm'].feature_names_in_
            
            # Add missing columns with zeros
            for feature in required_features:
                if feature not in input_data_encoded.columns:
                    input_data_encoded[feature] = 0
                    
            # Reorder columns to match model's expectations
            input_data_encoded = input_data_encoded[required_features]
            
        return input_data_encoded
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        raise

def make_predictions(features):
    """Make predictions using all models"""
    try:
        predictions = []
        for model_name, model in MODELS.items():
            pred = model.predict(features)[0] if features.shape[0] == 1 else model.predict(features)
            predictions.append(pred)
        
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

@app.post("/predict_manual")
async def predict_manual(request: Request):
    try:
        data = await request.json()
        print("Received manual input data")
        
        # Prepare features
        features = prepare_features(data, is_manual=True)
        
        # Make predictions
        model_predictions = {
            'svm': MODELS['svm'].predict(features)[0],
            'decision_tree': MODELS['dt'].predict(features)[0],
            'naive_bayes': MODELS['nb'].predict(features)[0],
            'random_forest': MODELS['rf'].predict(features)[0]
        }
        
        # Get final prediction using mode
        from statistics import mode
        final_prediction = mode(list(model_predictions.values()))
        
        # Map predictions to labels
        result = PREDICTION_MAPPING[final_prediction]
        model_predictions = {k: PREDICTION_MAPPING[v] for k, v in model_predictions.items()}
        
        return {
            "prediction": result,
            "model_predictions": model_predictions
        }
        
    except Exception as e:
        print(f"Error in predict_manual: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Remove label column if it exists
        feature_df = df.drop(['label', 'Label', 'target', 'Target', 'class', 'Class'], 
                            axis=1, 
                            errors='ignore')
        
        # Prepare features
        features = prepare_features(feature_df, is_manual=False)

        # Make predictions
        predictions = make_predictions(features)
        
        # Add predictions to DataFrame
        df['SVM_Prediction'] = predictions[0]
        df['DT_Prediction'] = predictions[1]
        df['NB_Prediction'] = predictions[2]
        df['RF_Prediction'] = predictions[3]
        
        # Calculate final prediction
        predictions_only = df[['SVM_Prediction', 'DT_Prediction', 'NB_Prediction', 'RF_Prediction']]
        df['Final_Prediction'] = predictions_only.mode(axis=1)[0]
        
        # Map predictions to labels
        for col in ['SVM_Prediction', 'DT_Prediction', 'NB_Prediction', 'RF_Prediction', 'Final_Prediction']:
            df[col] = df[col].map(PREDICTION_MAPPING)
        
        # Save results
        timestamp = int(time.time())
        output_filename = f"predicted_dataset_{timestamp}.csv"
        output_path = UPLOAD_DIR / output_filename
        df.to_csv(output_path, index=False)
        
        # Calculate summary
        summary_stats = {
            'total': int(len(df)),
            'normal': int((df['Final_Prediction'] == 'Normal').sum()),
            'ddos': int((df['Final_Prediction'] == 'DDoS Attack').sum()),
            'probe': int((df['Final_Prediction'] == 'Probe Attack').sum())
        }
        
        return {
            "summary": summary_stats,
            "download_path": output_filename,
            "file_id": timestamp
        }
        
    except Exception as e:
        print(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_predictions/{file_id}")
async def get_predictions(
    file_id: str,
    start: int = Query(0),
    length: int = Query(10),
    search: str = Query(None),
    order_column: int = Query(0),
    order_dir: str = Query("asc")
):
    try:
        file_path = next(UPLOAD_DIR.glob(f"predicted_dataset_{file_id}.csv"))
        df = pd.read_csv(file_path)
        
        total_records = len(df)
        
        if search:
            df = df[df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        
        filtered_records = len(df)
        
        if order_column is not None:
            columns = df.columns
            if 0 <= order_column < len(columns):
                ascending = order_dir.lower() == "asc"
                df = df.sort_values(columns[order_column], ascending=ascending)
        
        df = df.iloc[start:start + length]
        
        return {
            "draw": 1,
            "recordsTotal": total_records,
            "recordsFiltered": filtered_records,
            "data": df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )

@app.get("/", response_class=HTMLResponse)
async def render_page(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html data-theme="light">
    <head>
        <title>DDoS Detection System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --success: #2ec4b6;
                --danger: #e63946;
                --warning: #ff9f1c;
                --info: #4cc9f0;
            }

            [data-theme="light"] {
                --bg-main: #f6f8fd;
                --bg-card: #ffffff;
                --text-main: #2d3748;
                --text-muted: #718096;
                --border-color: #e2e8f0;
                --shadow-color: rgba(0, 0, 0, 0.1);
            }

            [data-theme="dark"] {
                --bg-main: #1a202c;
                --bg-card: #2d3748;
                --text-main: #f7fafc;
                --text-muted: #a0aec0;
                --border-color: #4a5568;
                --shadow-color: rgba(0, 0, 0, 0.3);
            }

            body { 
                font-family: 'Poppins', sans-serif;
                background: var(--bg-main);
                color: var(--text-main);
                min-height: 100vh;
                padding: 2rem 1rem;
                transition: all 0.3s ease;
            }

            .container {
                background: var(--bg-card);
                border-radius: 20px;
                box-shadow: 0 10px 30px var(--shadow-color);
                padding: 2rem;
            }

            /* Theme Toggle Switch */
            .theme-switch {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
            }

            .theme-switch-button {
                background: var(--bg-card);
                border: 2px solid var(--border-color);
                color: var(--text-main);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .theme-switch-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px var(--shadow-color);
            }

            .header {
                text-align: center;
                margin-bottom: 3rem;
                padding: 2rem;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                border-radius: 15px;
                color: white;
            }

            .card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                box-shadow: 0 5px 15px var(--shadow-color);
                transition: all 0.3s ease;
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px var(--shadow-color);
            }

            .nav-tabs {
                border: none;
                margin-bottom: 2rem;
                gap: 1rem;
            }

            .nav-tabs .nav-link {
                color: var(--text-main);
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 10px;
                padding: 1rem 2rem;
                transition: all 0.3s ease;
            }

            .nav-tabs .nav-link.active {
                background: var(--primary);
                color: white;
                border-color: var(--primary);
            }

            .form-control, .form-select {
                background: var(--bg-card);
                border: 2px solid var(--border-color);
                color: var(--text-main);
                border-radius: 10px;
            }

            .form-control:focus, .form-select:focus {
                background: var(--bg-card);
                color: var(--text-main);
                border-color: var(--primary);
                box-shadow: 0 0 0 0.25rem rgba(67, 97, 238, 0.25);
            }

            .input-group-text {
                background: var(--bg-card);
                border: 2px solid var(--border-color);
                color: var(--text-main);
            }

            .table {
                color: var(--text-main);
            }

            .table thead th {
                background: var(--bg-main);
                border-bottom: none;
            }

            .stats-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
            }

            .stats-card h5 {
                color: var(--text-main);
            }

            /* Add your existing styles here */
            
        </style>
    </head>
    <body>
        <!-- Theme Toggle Switch -->
        <div class="theme-switch">
            <button class="theme-switch-button" id="themeToggle">
                <i class="fas fa-sun"></i>
                <span class="ms-2">Toggle Theme</span>
            </button>
        </div>

        <div class="container">
            <div class="text-center mb-5">
                <h1 class="display-4 fw-bold text-primary mb-3">DDoS Detection System</h1>
                <p class="lead text-muted">Analyze network traffic for potential DDoS attacks</p>
            </div>
            
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item">
                    <a class="nav-link active" id="manual-tab" data-bs-toggle="tab" href="#manual">Manual Input</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" id="file-tab" data-bs-toggle="tab" href="#file">File Upload</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/packet-capture" target="_blank">
                        <i class="fas fa-network-wired me-1"></i>Packet Capture
                    </a>
                </li>
            </ul>
            
            <div class="tab-content mt-4" id="myTabContent">
                <!-- Manual Input Tab -->
                <div class="tab-pane fade show active" id="manual">
                    <div class="card p-4 mb-4">
                        <form id="manualForm" class="needs-validation" novalidate>
                            <div class="row g-4">
                                <div class="col-md-4">
                                    <div class="form-floating">
                                        <input type="number" class="form-control" id="duration" name="duration" value="0" required>
                                        <label for="duration">Duration</label>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="form-floating">
                                        <select class="form-control" id="protocol_type" name="protocol_type" required>
                                            <option value="tcp">TCP</option>
                                            <option value="udp">UDP</option>
                                            <option value="icmp">ICMP</option>
                                        </select>
                                        <label for="protocol_type">Protocol Type</label>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="form-floating">
                                        <select class="form-control" id="service" name="service" required>
                                            <option value="http">HTTP</option>
                                            <option value="ftp">FTP</option>
                                            <option value="smtp">SMTP</option>
                                            <option value="ssh">SSH</option>
                                            <option value="dns">DNS</option>
                                        </select>
                                        <label for="service">Service</label>
                                    </div>
                                </div>
                            </div>

                            <div class="row g-4 mt-2">
                                <div class="col-md-4">
                                    <div class="form-floating">
                                        <input type="number" class="form-control" id="src_bytes" name="src_bytes" value="0" required>
                                        <label for="src_bytes">Source Bytes</label>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="form-floating">
                                        <input type="number" class="form-control" id="dst_bytes" name="dst_bytes" value="0" required>
                                        <label for="dst_bytes">Destination Bytes</label>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="form-floating">
                                        <input type="number" class="form-control" id="count" name="count" value="0" required>
                                        <label for="count">Count</label>
                                    </div>
                                </div>
                            </div>

                            <div class="d-grid gap-2 d-md-flex justify-content-md-end mt-4">
                                <button type="submit" class="btn btn-primary btn-lg" id="predictBtn">
                                    <span class="spinner-border spinner-border-sm d-none me-2" role="status"></span>
                                    Analyze Traffic
                                </button>
                            </div>
                        </form>
                    </div>

                    <div id="manualResult" class="mt-4 fade-in" style="display:none">
                        <div class="card">
                            <div class="card-body">
                                <h4 class="card-title mb-4">Analysis Results</h4>
                                <div id="predictionText" class="display-5 mb-4 text-center"></div>
                                <div class="row">
                                    <div class="col-md-8 mx-auto">
                                        <div id="modelPredictions" class="d-flex flex-column gap-3"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- File Upload Tab -->
                <div class="tab-pane fade" id="file">
                    <div class="card p-4">
                        <form id="uploadForm">
                            <div class="upload-area mb-4">
                                <i class="fas fa-cloud-upload-alt fa-3x text-primary mb-3"></i>
                            <h4>Upload CSV File for Analysis</h4>
                            <p class="text-muted">Drag and drop your file here or click to browse</p>
                                <input type="file" class="form-control" name="file" accept=".csv" required>
                        </div>
                            <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                                <button type="submit" class="btn btn-primary btn-lg">
                                    <span class="spinner-border spinner-border-sm d-none me-2" role="status"></span>
                                    Analyze File
                            </button>
                        </div>
                    </form>
                    </div>
                    
                    <div id="fileResult" class="mt-4 fade-in" style="display:none">
                        <div class="row g-4">
                            <div class="col-md-3">
                                <div class="card h-100 text-center">
                                    <div class="card-body">
                                        <h5 class="card-title text-muted mb-3">Normal Traffic</h5>
                                        <p class="display-6 text-success mb-0" id="normal-count">0</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card h-100 text-center">
                                    <div class="card-body">
                                        <h5 class="card-title text-muted mb-3">DDoS Attacks</h5>
                                        <p class="display-6 text-danger mb-0" id="ddos-count">0</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card h-100 text-center">
                                    <div class="card-body">
                                        <h5 class="card-title text-muted mb-3">Probe Attacks</h5>
                                        <p class="display-6 text-warning mb-0" id="probe-count">0</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card h-100 text-center">
                                    <div class="card-body">
                                        <h5 class="card-title text-muted mb-3">Total Records</h5>
                                        <p class="display-6 text-info mb-0" id="total-count">0</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                        <div class="card mt-4">
                            <div class="card-body">
                        <div class="table-responsive">
                            <table id="resultsTable" class="table table-striped">
                                        <thead><tr></tr></thead>
                                <tbody></tbody>
                            </table>
                                </div>
                        </div>
                    </div>

                        <div class="text-center mt-4">
                            <button id="downloadBtn" class="btn btn-success btn-lg">
                                <i class="fas fa-download me-2"></i>Download Complete Dataset
                        </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Theme Toggle Function
            const themeToggle = document.getElementById('themeToggle');
            const html = document.documentElement;
            const icon = themeToggle.querySelector('i');

            // Check for saved theme preference
            const savedTheme = localStorage.getItem('theme') || 'light';
            html.setAttribute('data-theme', savedTheme);
            updateThemeIcon(savedTheme);

            themeToggle.addEventListener('click', () => {
                const currentTheme = html.getAttribute('data-theme');
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                
                html.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                updateThemeIcon(newTheme);
            });

            function updateThemeIcon(theme) {
                icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
            }

            $(document).ready(function() {
                // Manual Form Handler
                $('#manualForm').on('submit', function(e) {
                    e.preventDefault();
                    
                    const $btn = $('#predictBtn');
                    const $spinner = $btn.find('.spinner-border');
                    $btn.prop('disabled', true);
                    $spinner.removeClass('d-none');
                    
                    const formData = {};
                    $(this).serializeArray().forEach(item => {
                        formData[item.name] = item.value;
                    });
                    
                    $.ajax({
                        url: '/predict_manual',
                        method: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify(formData),
                        success: function(response) {
                            $('#manualResult').show();
                            
                            // Set prediction text with appropriate color
                            const predClass = response.prediction === 'DDoS Attack' ? 'text-danger' :
                                            response.prediction === 'Normal' ? 'text-success' : 'text-warning';
                            $('#predictionText').attr('class', `h3 mb-4 ${predClass}`).text(response.prediction);
                            
                            // Show individual model predictions
                            const predictions = Object.entries(response.model_predictions)
                                .map(([model, pred]) => {
                                    const badgeClass = pred === 'DDoS Attack' ? 'bg-danger' :
                                                     pred === 'Normal' ? 'bg-success' : 'bg-warning';
                                    return `
                                        <div class="mb-2">
                                            <strong>${model.toUpperCase()}:</strong>
                                            <span class="badge ${badgeClass}">${pred}</span>
                        </div>
                    `;
                                }).join('');
                            
                            $('#modelPredictions').html(predictions);
                        },
                        error: function(xhr, status, error) {
                            $('#manualResult').show();
                            $('#predictionText').attr('class', 'h3 mb-4 text-danger').text('Error: ' + error);
                            $('#modelPredictions').empty();
                        },
                        complete: function() {
                            $btn.prop('disabled', false);
                            $spinner.addClass('d-none');
                        }
                    });
                });

                // File Upload Handler
                $('#uploadForm').on('submit', function(e) {
                    e.preventDefault();
                    
                    const $btn = $(this).find('button[type="submit"]');
                    const $spinner = $btn.find('.spinner-border');
                    $btn.prop('disabled', true);
                    $spinner.removeClass('d-none');
                    
                    const formData = new FormData(this);
                    
                    $.ajax({
                        url: '/predict',
                        method: 'POST',
                        data: formData,
                        processData: false,
                        contentType: false,
                        success: function(response) {
                            $('#fileResult').show();
                            
                            // Update summary counts
                            $('#normal-count').text(response.summary.normal);
                            $('#ddos-count').text(response.summary.ddos);
                            $('#probe-count').text(response.summary.probe);
                            $('#total-count').text(response.summary.total);
                            
                            // Initialize DataTable
                    if ($.fn.DataTable.isDataTable('#resultsTable')) {
                        $('#resultsTable').DataTable().destroy();
                    }
                    
                    $('#resultsTable').DataTable({
                        processing: true,
                        serverSide: true,
                                ajax: `/get_predictions/${response.file_id}`,
                        columns: [
                            { data: 'duration', title: 'Duration' },
                                    { data: 'protocol_type', title: 'Protocol' },
                                    { data: 'service', title: 'Service' },
                                    { data: 'src_bytes', title: 'Source Bytes' },
                                    { data: 'dst_bytes', title: 'Dest Bytes' },
                            { 
                                data: 'Final_Prediction', 
                                        title: 'Prediction',
                                render: function(data) {
                                            const className = data === 'DDoS Attack' ? 'attack' :
                                                            data === 'Normal' ? 'normal' : 'probe';
                                            return `<span class="prediction-cell ${className}">${data}</span>`;
                                        }
                                    }
                                ]
                            });
                            
                            // Setup download button
                            $('#downloadBtn').off('click').on('click', function() {
                                window.location.href = `/download/${response.download_path}`;
                            });
                        },
                        error: function(xhr, status, error) {
                            alert('Error processing file: ' + error);
                        },
                        complete: function() {
                            $btn.prop('disabled', false);
                            $spinner.addClass('d-none');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def process_packet(packet):
    """Process captured packet and update statistics with DDoS detection"""
    try:
        with PACKET_LOCK:
            if IP in packet:
                ip = packet[IP]
                src_ip = ip.src
                
                # Initialize start time if not set
                if PACKET_BUFFER[src_ip]['start_time'] is None:
                    PACKET_BUFFER[src_ip]['start_time'] = time.time()
                    PACKET_BUFFER[src_ip]['detection_results'] = {}
                    PACKET_BUFFER[src_ip]['final_prediction'] = 'Normal'
                    PACKET_BUFFER[src_ip]['hot'] = 0
                    PACKET_BUFFER[src_ip]['num_failed_logins'] = 0
                    PACKET_BUFFER[src_ip]['logged_in'] = 0
                    PACKET_BUFFER[src_ip]['num_compromised'] = 0
                    PACKET_BUFFER[src_ip]['root_shell'] = 0
                    PACKET_BUFFER[src_ip]['su_attempted'] = 0
                    PACKET_BUFFER[src_ip]['num_root'] = 0
                    PACKET_BUFFER[src_ip]['num_file_creations'] = 0
                    PACKET_BUFFER[src_ip]['num_shells'] = 0
                    PACKET_BUFFER[src_ip]['num_access_files'] = 0
                    PACKET_BUFFER[src_ip]['num_outbound_cmds'] = 0
                    PACKET_BUFFER[src_ip]['is_host_login'] = 0
                    PACKET_BUFFER[src_ip]['is_guest_login'] = 0
                
                # Update basic metrics
                PACKET_BUFFER[src_ip]['packets'] += 1
                PACKET_BUFFER[src_ip]['bytes'] += len(packet)
                PACKET_BUFFER[src_ip]['unique_ips'].add(ip.dst)
                
                # Identify protocol and update protocol-specific metrics
                if TCP in packet:
                    PACKET_BUFFER[src_ip]['protocols']['TCP'] += 1
                    PACKET_BUFFER[src_ip]['dest_ports'][packet[TCP].dport] += 1
                    PACKET_BUFFER[src_ip]['src_ports'][packet[TCP].sport] += 1
                    # Check for potential SYN flood
                    if packet[TCP].flags & 0x02:  # SYN flag
                        PACKET_BUFFER[src_ip]['hot'] += 1
                elif UDP in packet:
                    PACKET_BUFFER[src_ip]['protocols']['UDP'] += 1
                    PACKET_BUFFER[src_ip]['dest_ports'][packet[UDP].dport] += 1
                    PACKET_BUFFER[src_ip]['src_ports'][packet[UDP].sport] += 1
                elif ICMP in packet:
                    PACKET_BUFFER[src_ip]['protocols']['ICMP'] += 1
                
                # Perform DDoS detection
                current_time = time.time()
                duration = current_time - PACKET_BUFFER[src_ip]['start_time']
                if duration > 0:
                    # Calculate features for detection using the same logic as file upload and manual input
                    features = {
                        'duration': duration,
                        'protocol_type': 'tcp' if TCP in packet else 'udp' if UDP in packet else 'icmp',
                        'service': 'http' if TCP in packet and packet[TCP].dport == 80 else 'other',
                        'flag': 'SF',  # Simplified flag
                        'src_bytes': PACKET_BUFFER[src_ip]['bytes'],
                        'dst_bytes': len(packet),
                        'land': 1 if ip.src == ip.dst else 0,
                        'wrong_fragment': 0,  # Simplified
                        'urgent': 0,  # Simplified
                        'hot': PACKET_BUFFER[src_ip]['hot'],
                        'num_failed_logins': PACKET_BUFFER[src_ip]['num_failed_logins'],
                        'logged_in': PACKET_BUFFER[src_ip]['logged_in'],
                        'num_compromised': PACKET_BUFFER[src_ip]['num_compromised'],
                        'root_shell': PACKET_BUFFER[src_ip]['root_shell'],
                        'su_attempted': PACKET_BUFFER[src_ip]['su_attempted'],
                        'num_root': PACKET_BUFFER[src_ip]['num_root'],
                        'num_file_creations': PACKET_BUFFER[src_ip]['num_file_creations'],
                        'num_shells': PACKET_BUFFER[src_ip]['num_shells'],
                        'num_access_files': PACKET_BUFFER[src_ip]['num_access_files'],
                        'num_outbound_cmds': PACKET_BUFFER[src_ip]['num_outbound_cmds'],
                        'is_host_login': PACKET_BUFFER[src_ip]['is_host_login'],
                        'is_guest_login': PACKET_BUFFER[src_ip]['is_guest_login'],
                        'count': PACKET_BUFFER[src_ip]['packets'],
                        'srv_count': len(PACKET_BUFFER[src_ip]['src_ports']),
                        'serror_rate': PACKET_BUFFER[src_ip]['hot'] / PACKET_BUFFER[src_ip]['packets'],
                        'srv_serror_rate': 0,  # Simplified
                        'rerror_rate': 0,  # Simplified
                        'srv_rerror_rate': 0,  # Simplified
                        'same_srv_rate': 1 if len(PACKET_BUFFER[src_ip]['dest_ports']) == 1 else 0,
                        'diff_srv_rate': 0 if len(PACKET_BUFFER[src_ip]['dest_ports']) == 1 else 1,
                        'srv_diff_host_rate': len(PACKET_BUFFER[src_ip]['unique_ips']) / PACKET_BUFFER[src_ip]['packets'],
                        'dst_host_count': len(PACKET_BUFFER[src_ip]['unique_ips']),
                        'dst_host_srv_count': len(PACKET_BUFFER[src_ip]['dest_ports']),
                        'dst_host_same_srv_rate': 1 if len(PACKET_BUFFER[src_ip]['dest_ports']) == 1 else 0,
                        'dst_host_diff_srv_rate': 0 if len(PACKET_BUFFER[src_ip]['dest_ports']) == 1 else 1,
                        'dst_host_same_src_port_rate': 1 if len(PACKET_BUFFER[src_ip]['src_ports']) == 1 else 0,
                        'dst_host_srv_diff_host_rate': len(PACKET_BUFFER[src_ip]['unique_ips']) / max(len(PACKET_BUFFER[src_ip]['dest_ports']), 1),
                        'dst_host_serror_rate': PACKET_BUFFER[src_ip]['hot'] / PACKET_BUFFER[src_ip]['packets'],
                        'dst_host_srv_serror_rate': 0,  # Simplified
                        'dst_host_rerror_rate': 0,  # Simplified
                        'dst_host_srv_rerror_rate': 0  # Simplified
                    }
                    
                    # Prepare features for model using the same function as file upload and manual input
                    features_df = pd.DataFrame([features])
                    features_prepared = prepare_features(features_df, is_manual=False)
                    
                    # Make predictions using the same models and weighting as file upload and manual input
                    predictions = make_predictions(features_prepared)
                    weights = {'svm': 0.3, 'dt': 0.2, 'nb': 0.2, 'rf': 0.3}  # Weights based on model performance
                    
                    # Calculate weighted predictions
                    weighted_predictions = []
                    detection_results = {}
                    
                    for i, (model_name, weight) in enumerate(weights.items()):
                        pred = predictions[i] if isinstance(predictions[i], (int, float)) else predictions[i][0]
                        weighted_predictions.append(pred * weight)
                        detection_results[model_name] = PREDICTION_MAPPING[pred]
                    
                    PACKET_BUFFER[src_ip]['detection_results'] = detection_results
                    
                    # Calculate final prediction with confidence threshold
                    weighted_sum = sum(weighted_predictions)
                    confidence_threshold = 0.6
                    final_prediction = 1 if weighted_sum > confidence_threshold else 0
                    PACKET_BUFFER[src_ip]['final_prediction'] = PREDICTION_MAPPING[final_prediction]
                    
                    # Reset metrics if duration exceeds window
                    if duration > 60:  # Reset after 1 minute
                        PACKET_BUFFER[src_ip]['start_time'] = current_time
                        PACKET_BUFFER[src_ip]['packets'] = 0
                        PACKET_BUFFER[src_ip]['bytes'] = 0
                        PACKET_BUFFER[src_ip]['hot'] = 0
                        PACKET_BUFFER[src_ip]['unique_ips'].clear()
                        PACKET_BUFFER[src_ip]['protocols'].clear()
                        PACKET_BUFFER[src_ip]['dest_ports'].clear()
                        PACKET_BUFFER[src_ip]['src_ports'].clear()
    except Exception as e:
        print(f"Error processing packet: {e}")
        return None
    return PACKET_BUFFER[src_ip]

def capture_packets(interface):
    """Start packet capture on specified interface with enhanced feature extraction"""
    global IS_CAPTURING, PACKET_BUFFER
    try:
        IS_CAPTURING = True
        # Initialize sliding window parameters
        window_size = 100  # Number of packets to analyze together
        window_data = []
        start_time = time.time()
        
        def process_window(packets):
            if len(packets) < window_size:
                return
            
            # Extract features similar to training data
            features = {
                'duration': time.time() - start_time,
                'protocol_type': packets[0].get('protocol', 'tcp'),
                'service': 'http',  # Default to http, can be enhanced
                'flag': 'SF',  # Default to SF (normal connection)
                'src_bytes': sum(p.get('bytes', 0) for p in packets),
                'dst_bytes': sum(p.get('dst_bytes', 0) for p in packets),
                'land': 0,
                'wrong_fragment': sum(1 for p in packets if p.get('wrong_fragment', 0)),
                'urgent': sum(1 for p in packets if p.get('urgent', 0)),
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': len(packets),
                'srv_count': sum(1 for p in packets if p.get('service') == packets[0].get('service')),
                'serror_rate': sum(1 for p in packets if p.get('error', False)) / len(packets),
                'srv_serror_rate': 0,
                'rerror_rate': 0,
                'srv_rerror_rate': 0,
                'same_srv_rate': sum(1 for p in packets if p.get('service') == packets[0].get('service')) / len(packets),
                'diff_srv_rate': sum(1 for p in packets if p.get('service') != packets[0].get('service')) / len(packets),
                'srv_diff_host_rate': 0,
                'dst_host_count': len(set(p.get('dst_ip') for p in packets)),
                'dst_host_srv_count': 0,
                'dst_host_same_srv_rate': 0,
                'dst_host_diff_srv_rate': 0,
                'dst_host_same_src_port_rate': 0,
                'dst_host_srv_diff_host_rate': 0,
                'dst_host_serror_rate': 0,
                'dst_host_srv_serror_rate': 0,
                'dst_host_rerror_rate': 0,
                'dst_host_srv_rerror_rate': 0
            }
            
            # Prepare features for prediction
            prepared_features = prepare_features(features)
            
            # Make predictions with confidence weighting
            predictions = []
            weights = {'svm': 0.3, 'dt': 0.2, 'nb': 0.2, 'rf': 0.3}  # Weights based on model performance
            
            for model_name, model in MODELS.items():
                pred = model.predict(prepared_features)[0]
                confidence = weights[model_name]
                predictions.append((pred, confidence))
            
            # Calculate weighted prediction
            weighted_sum = sum(pred * conf for pred, conf in predictions)
            threshold = 0.6  # Confidence threshold for attack detection
            
            final_prediction = 1 if weighted_sum > threshold else 0
            
            # Update packet buffer with prediction
            with PACKET_LOCK:
                PACKET_BUFFER['prediction'] = PREDICTION_MAPPING[final_prediction]
                PACKET_BUFFER['confidence'] = weighted_sum
            
            # Clear window for next batch
            window_data.clear()
        
        def enhanced_packet_processor(packet):
            packet_info = process_packet(packet)
            window_data.append(packet_info)
            
            if len(window_data) >= window_size:
                process_window(window_data)
        
        # Start capture with enhanced processing
        sniff(iface=interface,
              prn=enhanced_packet_processor,
              store=0,
              stop_filter=lambda _: not IS_CAPTURING,
              filter="ip")
    except Exception as e:
        print(f"Error in packet capture: {e}")
        IS_CAPTURING = False

def get_interface_stats(ip_address):
    """Get statistics for specific IP address"""
    with PACKET_LOCK:
        stats = PACKET_BUFFER.get(ip_address, None)
        if not stats:
            return None
        
        current_time = time.time()
        duration = current_time - stats['start_time'] if stats['start_time'] else 0
        
        if duration == 0:
            duration = 1  # Avoid division by zero
        
        return {
            'duration': duration,
            'packets_per_second': stats['packets'] / duration,
            'bytes_per_second': stats['bytes'] / duration,
            'total_packets': stats['packets'],
            'total_bytes': stats['bytes'],
            'unique_destinations': len(stats['unique_ips']),
            'detection_results': stats.get('detection_results', {}),
            'final_prediction': stats.get('final_prediction', 'Normal'),
            'protocols': dict(stats['protocols']),
            'top_dest_ports': dict(sorted(stats['dest_ports'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]),
            'timestamp': current_time
        }

@app.post("/capture/start/{interface}")
async def start_capture(interface: str):
    global CAPTURE_THREAD
    try:
        if IS_CAPTURING:
            return {"status": "already_running"}
        
        CAPTURE_THREAD = Thread(target=capture_packets, args=(interface,))
        CAPTURE_THREAD.daemon = True
        CAPTURE_THREAD.start()
        
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/capture/stop")
async def stop_capture():
    global IS_CAPTURING, CAPTURE_THREAD
    try:
        IS_CAPTURING = False
        if CAPTURE_THREAD:
            CAPTURE_THREAD.join(timeout=2)
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capture/stats/{ip_address}")
async def get_capture_stats(ip_address: str):
    try:
        stats = get_interface_stats(ip_address)
        if not stats:
            return {"status": "no_data"}
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/packet-capture", response_class=HTMLResponse)
async def packet_capture_page(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html data-theme="light">
    <head>
        <title>Packet Capture DDoS Detection</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --success: #2ec4b6;
                --danger: #e63946;
                --warning: #ff9f1c;
                --info: #4cc9f0;
            }

            [data-theme="light"] {
                --bg-main: #f6f8fd;
                --bg-card: #ffffff;
                --text-main: #2d3748;
                --text-muted: #718096;
                --border-color: #e2e8f0;
                --shadow-color: rgba(0, 0, 0, 0.1);
            }

            [data-theme="dark"] {
                --bg-main: #1a202c;
                --bg-card: #2d3748;
                --text-main: #f7fafc;
                --text-muted: #a0aec0;
                --border-color: #4a5568;
                --shadow-color: rgba(0, 0, 0, 0.3);
            }

            body { 
                font-family: 'Poppins', sans-serif;
                background: var(--bg-main);
                color: var(--text-main);
                min-height: 100vh;
                padding: 2rem 1rem;
                transition: all 0.3s ease;
            }

            .container {
                background: var(--bg-card);
                border-radius: 20px;
                box-shadow: 0 10px 30px var(--shadow-color);
                padding: 2rem;
            }

            .capture-status {
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
            }

            .status-active {
                background-color: var(--success);
                color: white;
            }

            .status-inactive {
                background-color: var(--danger);
                color: white;
            }

            .metric-card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
            }

            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px var(--shadow-color);
            }

            .chart-container {
                height: 300px;
                margin-bottom: 1rem;
                background: var(--bg-card);
                border-radius: 15px;
                padding: 1rem;
            }

            .card {
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                box-shadow: 0 5px 15px var(--shadow-color);
                transition: all 0.3s ease;
                margin-bottom: 1.5rem;
            }

            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px var(--shadow-color);
            }

            .form-control, .form-select {
                background: var(--bg-card);
                border: 2px solid var(--border-color);
                color: var(--text-main);
                border-radius: 10px;
                padding: 0.75rem 1rem;
            }

            .form-control:focus, .form-select:focus {
                background: var(--bg-card);
                color: var(--text-main);
                border-color: var(--primary);
                box-shadow: 0 0 0 0.25rem rgba(67, 97, 238, 0.25);
            }

            .btn-primary {
                background: var(--primary);
                border-color: var(--primary);
                border-radius: 10px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }

            .btn-primary:hover {
                background: var(--secondary);
                border-color: var(--secondary);
                transform: translateY(-2px);
            }

            .btn-danger {
                background: var(--danger);
                border-color: var(--danger);
                border-radius: 10px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }

            .btn-danger:hover {
                background: #dc2f3f;
                border-color: #dc2f3f;
                transform: translateY(-2px);
            }

            .alert {
                border-radius: 10px;
                padding: 1rem 1.5rem;
                margin-bottom: 1.5rem;
                border: none;
                transition: all 0.3s ease;
            }

            /* Theme Switch Button */
            .theme-switch {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
            }

            .theme-switch-button {
                background: var(--bg-card);
                border: 2px solid var(--border-color);
                color: var(--text-main);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .theme-switch-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px var(--shadow-color);
            }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const themeToggle = document.getElementById('themeToggle');
                const html = document.documentElement;
                const icon = themeToggle.querySelector('i');

                // Check for saved theme preference
                const savedTheme = localStorage.getItem('theme');
                if (savedTheme) {
                    html.setAttribute('data-theme', savedTheme);
                    icon.className = savedTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
                }

                // Theme toggle functionality
                themeToggle.addEventListener('click', function() {
                    const currentTheme = html.getAttribute('data-theme');
                    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                    
                    html.setAttribute('data-theme', newTheme);
                    localStorage.setItem('theme', newTheme);
                    icon.className = newTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
                });
            });
        </script>
    </head>
    <body>
        <!-- Theme Toggle Switch -->
        <div class="theme-switch">
            <button class="theme-switch-button" id="themeToggle">
                <i class="fas fa-sun"></i>
                <span class="ms-2">Toggle Theme</span>
            </button>
        </div>
        <div class="container mt-4">
            <div class="row mb-4">
                <div class="col-12">
                    <h1 class="text-center">Packet Capture DDoS Detection</h1>
                    <p class="text-center text-muted">Real-time network traffic analysis</p>
                </div>
            </div>

            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Capture Settings</h5>
                            <div class="mb-3">
                                <label for="interface" class="form-label">Network Interface</label>
                                <select class="form-control" id="interface">
                                    <option value="">Loading interfaces...</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label for="ipAddress" class="form-label">Target IP Address</label>
                                <input type="text" class="form-control" id="ipAddress" 
                                       placeholder="Enter IP to monitor">
                            </div>
                            <div class="d-grid gap-2">
                                <button class="btn btn-primary" id="startCapture">
                                    <i class="fas fa-play me-2"></i>Start Capture
                                </button>
                                <button class="btn btn-danger" id="stopCapture" style="display:none">
                                    <i class="fas fa-stop me-2"></i>Stop Capture
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Capture Status</h5>
                            <div id="captureStatus" class="capture-status status-inactive">
                                Capture Inactive
                            </div>
                            <div id="captureMetrics">
                                <div class="metric-card">
                                    <h6>Traffic Rate</h6>
                                    <p class="mb-0">
                                        Packets/s: <span id="packetsPerSecond">0</span><br>
                                        Bytes/s: <span id="bytesPerSecond">0</span>
                                    </p>
                                </div>
                                <div class="metric-card">
                                    <h6>Total Traffic</h6>
                                    <p class="mb-0">
                                        Packets: <span id="totalPackets">0</span><br>
                                        Bytes: <span id="totalBytes">0</span>
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Traffic Rate</h5>
                            <div class="chart-container">
                                <canvas id="trafficChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Protocol Distribution</h5>
                            <div class="chart-container">
                                <canvas id="protocolChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Detection Results</h5>
                            <div class="alert" id="detectionAlert" role="alert"></div>
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h6>SVM Prediction</h6>
                                        <p id="svmPrediction" class="mb-0">Normal</p>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h6>Decision Tree Prediction</h6>
                                        <p id="dtPrediction" class="mb-0">Normal</p>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h6>Naive Bayes Prediction</h6>
                                        <p id="nbPrediction" class="mb-0">Normal</p>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="metric-card">
                                        <h6>Random Forest Prediction</h6>
                                        <p id="rfPrediction" class="mb-0">Normal</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Initialize charts
            const trafficCtx = document.getElementById('trafficChart').getContext('2d');
            const protocolCtx = document.getElementById('protocolChart').getContext('2d');

            const trafficChart = new Chart(trafficCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Packets/s',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });

            const protocolChart = new Chart(protocolCtx, {
                type: 'pie',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            'rgb(255, 99, 132)',
                            'rgb(54, 162, 235)',
                            'rgb(255, 205, 86)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });

            // Update interface list
            $.get('/get_interfaces', function(data) {
                const select = $('#interface');
                select.empty();
                
                if (data.interfaces && data.interfaces.length > 0) {
                    data.interfaces.forEach(interface => {
                        select.append(`<option value="${interface.value}">${interface.description}</option>`);
                    });
                } else {
                    select.append(`<option value="">No interfaces found</option>`);
                }
            }).fail(function(error) {
                console.error('Error fetching interfaces:', error);
                const select = $('#interface');
                select.empty();
                select.append(`<option value="">Error loading interfaces</option>`);
            });

            let updateInterval;

            // Start capture
            $('#startCapture').click(function() {
                const interface = $('#interface').val();
                const ipAddress = $('#ipAddress').val();

                if (!interface || !ipAddress) {
                    alert('Please select an interface and enter an IP address');
                    return;
                }

                $.post(`/capture/start/${interface}`, function(response) {
                    if (response.status === 'started') {
                        $('#startCapture').hide();
                        $('#stopCapture').show();
                        $('#captureStatus')
                            .removeClass('status-inactive')
                            .addClass('status-active')
                            .text('Capture Active');

                        // Start updating stats
                        updateInterval = setInterval(() => {
                            $.get(`/capture/stats/${ipAddress}`, function(stats) {
                                if (stats.status !== 'no_data') {
                                    updateStats(stats);
                                }
                            });
                        }, 1000);
                    }
                });
            });

            // Stop capture
            $('#stopCapture').click(function() {
                $.post('/capture/stop', function(response) {
                    if (response.status === 'stopped') {
                        $('#stopCapture').hide();
                        $('#startCapture').show();
                        $('#captureStatus')
                            .removeClass('status-active')
                            .addClass('status-inactive')
                            .text('Capture Inactive');
                        
                        clearInterval(updateInterval);
                    }
                });
            });

            function updateStats(stats) {
                // Update metrics
                $('#packetsPerSecond').text(stats.packets_per_second.toFixed(2));
                $('#bytesPerSecond').text(formatBytes(stats.bytes_per_second));
                $('#totalPackets').text(stats.total_packets);
                $('#totalBytes').text(formatBytes(stats.total_bytes));

                // Update traffic chart
                if (trafficChart.data.labels.length > 20) {
                    trafficChart.data.labels.shift();
                    trafficChart.data.datasets[0].data.shift();
                }

                trafficChart.data.labels.push(new Date().toLocaleTimeString());
                trafficChart.data.datasets[0].data.push(stats.packets_per_second);
                trafficChart.update();

                // Update protocol chart
                protocolChart.data.labels = Object.keys(stats.protocols);
                protocolChart.data.datasets[0].data = Object.values(stats.protocols);
                protocolChart.update();

                // Update detection results
                if (stats.detection_results) {
                    $('#svmPrediction').text(stats.detection_results.svm);
                    $('#dtPrediction').text(stats.detection_results.dt);
                    $('#nbPrediction').text(stats.detection_results.nb);
                    $('#rfPrediction').text(stats.detection_results.rf);

                    // Update alert status
                    const alertDiv = $('#detectionAlert');
                    alertDiv.removeClass('alert-success alert-danger alert-warning');
                    
                    if (stats.final_prediction === 'DDoS Attack') {
                        alertDiv.addClass('alert-danger')
                            .text('⚠️ DDoS Attack Detected!');
                    } else if (stats.final_prediction === 'Probe Attack') {
                        alertDiv.addClass('alert-warning')
                            .text('⚠️ Probe Attack Detected!');
                    } else {
                        alertDiv.addClass('alert-success')
                            .text('✓ Normal Traffic');
                    }
                }
            }

            function formatBytes(bytes) {
                if (bytes < 1024) return bytes + " B";
                else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + " KB";
                else if (bytes < 1073741824) return (bytes / 1048576).toFixed(2) + " MB";
                else return (bytes / 1073741824).toFixed(2) + " GB";
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/get_interfaces")
async def get_interfaces():
    try:
        # Use only psutil for interface detection
        interfaces = []
        net_if_stats = psutil.net_if_stats()
        
        for iface, addrs in psutil.net_if_addrs().items():
            # Check if interface is up and not a loopback
            if (net_if_stats.get(iface) and 
                net_if_stats[iface].isup and 
                not iface.lower().startswith(('lo', 'loop'))):
                
                # Get MAC address if available
                mac = None
                for addr in addrs:
                    if addr.family == psutil.AF_LINK:
                        mac = addr.address
                        break
                
                # Get IPv4 address if available
                ipv4 = None
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ipv4 = addr.address
                        break
                
                description = f"{iface} ({ipv4 if ipv4 else 'No IP'}) - {mac if mac else 'No MAC'}"
                
                interfaces.append({
                    'name': iface,
                    'description': description,
                    'value': iface
                })
        
        # Sort interfaces by name
        interfaces.sort(key=lambda x: x['name'])
        
        print("Available interfaces:", interfaces)  # Debug print
        return {"interfaces": interfaces}
    except Exception as e:
        print(f"Error getting interfaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=5000)