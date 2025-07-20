#!/bin/bash

echo "🚀 Starting deployment process..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Make sure you're in the frontend directory."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build the application
echo "🔨 Building the application..."
npm run build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🎉 Ready for deployment!"
    echo ""
    echo "Next steps:"
    echo "1. Deploy backend to Railway/Render"
    echo "2. Update NEXT_PUBLIC_API_URL environment variable"
    echo "3. Deploy frontend to Vercel"
else
    echo "❌ Build failed. Please check the errors above."
    exit 1
fi