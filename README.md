# DDOS-Detection-using-ML-Techniques

A machine learning-based DDoS (Distributed Denial of Service) attack detection system with real-time packet capture capabilities and an interactive web interface.

## Features

- **Multi-Model Detection**: Utilizes multiple machine learning models (SVM, Decision Tree, Naive Bayes, Random Forest) for accurate attack detection
- **Real-time Packet Capture**: Live network traffic monitoring and analysis
- **Web Interface**: User-friendly interface for both manual input and file upload modes
- **Comprehensive Analysis**: Detects various types of attacks including DDoS and Probe attacks
- **Interactive Visualization**: Real-time traffic statistics and attack detection results

## Technical Architecture

- **Frontend**: HTML5, Bootstrap 5, DataTables for dynamic data display
- **Backend**: FastAPI for high-performance API endpoints
- **ML Models**: scikit-learn based models for attack detection
- **Network Analysis**: Scapy for packet capture and analysis

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python -m uvicorn endpoint:app --reload
```

2. Open your browser and navigate to `http://localhost:5000`

### Detection Modes

1. **Manual Input Mode**
   - Enter network traffic parameters manually
   - Get instant detection results

2. **File Upload Mode**
   - Upload CSV files containing network traffic data
   - Get batch processing results

3. **Real-time Packet Capture**
   - Monitor live network traffic
   - View real-time detection results

## Model Performance

The system uses a weighted ensemble of multiple models:
- Support Vector Machine (30% weight)
- Decision Tree (20% weight)
- Naive Bayes (20% weight)
- Random Forest (30% weight)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KDD Cup 1999 dataset for model training
- FastAPI framework for web implementation
- Scapy library for network packet analysis
