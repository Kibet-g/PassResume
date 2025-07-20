@echo off
echo 🚀 Setting up Open Resume Auditor...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 18+ first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Setup Backend
echo 📦 Setting up backend...
cd backend

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Install Python dependencies
pip install -r requirements.txt

REM Download spaCy model
python -m spacy download en_core_web_sm

echo ✅ Backend setup complete

REM Setup Frontend
echo 📦 Setting up frontend...
cd ..\frontend

REM Install Node.js dependencies
npm install

echo ✅ Frontend setup complete

REM Return to root directory
cd ..

echo 🎉 Setup complete!
echo.
echo To start the application:
echo 1. Start the backend:
echo    cd backend
echo    venv\Scripts\activate
echo    python app.py
echo.
echo 2. In a new terminal, start the frontend:
echo    cd frontend
echo    npm run dev
echo.
echo 3. Open http://localhost:3000 in your browser
echo.
echo Happy resume auditing! 🎯
pause