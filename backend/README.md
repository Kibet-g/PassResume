# Backend Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy English model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Run the Flask application:**
   ```bash
   python app.py
   ```

The backend server will start on `http://localhost:5000`

## API Endpoints

- `POST /api/upload` - Upload and parse resume files
- `POST /api/analyze` - Analyze resume against job description
- `GET /api/resume/{id}` - Get resume analysis results
- `GET /health` - Health check endpoint

## Database

The application uses SQLite database which will be automatically created when you first run the application. The database file `resume_auditor.db` will be created in the backend directory.

## Troubleshooting

1. **spaCy model not found:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Port already in use:**
   - Change the port in `app.py` (line with `app.run()`)
   - Or kill the process using the port

3. **Permission errors:**
   - Make sure you have write permissions in the backend directory
   - The app creates an `uploads` folder for temporary file storage