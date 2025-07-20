@echo off
echo 🚀 Starting deployment process...

REM Check if we're in the right directory
if not exist "package.json" (
    echo ❌ Error: package.json not found. Make sure you're in the frontend directory.
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
npm install

REM Build the application
echo 🔨 Building the application...
npm run build

REM Check if build was successful
if %errorlevel% equ 0 (
    echo ✅ Build successful!
    echo 🎉 Ready for deployment!
    echo.
    echo Next steps:
    echo 1. Deploy backend to Railway/Render
    echo 2. Update NEXT_PUBLIC_API_URL environment variable
    echo 3. Deploy frontend to Vercel
) else (
    echo ❌ Build failed. Please check the errors above.
    exit /b 1
)