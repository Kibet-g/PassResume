# ATS Resume Optimizer - Deployment Guide

## 🚀 Deployment Instructions

This project consists of a Next.js frontend and Flask backend that can be deployed separately.

### Frontend Deployment (Vercel)

1. **Deploy to Vercel:**
   ```bash
   npm install -g vercel
   vercel --prod
   ```

2. **Environment Variables:**
   Set the following environment variable in Vercel dashboard:
   - `NEXT_PUBLIC_API_URL`: Your backend URL (e.g., `https://your-app.railway.app`)

### Backend Deployment (Railway/Render)

#### Option 1: Railway
1. Visit [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Select the `backend` folder as the root directory
4. Railway will automatically detect the Flask app and deploy it

#### Option 2: Render
1. Visit [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Set the following:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python wsgi.py`

### Environment Variables for Backend

Set these environment variables in your deployment platform:

```
PORT=5000
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///resume_optimizer.db
```

### Post-Deployment Steps

1. **Update Frontend API URL:**
   - Update the API URL in your frontend code to point to your deployed backend
   - Set the `NEXT_PUBLIC_API_URL` environment variable in Vercel

2. **Test the Application:**
   - Verify user registration and login work
   - Test resume upload and analysis
   - Check the auto-improve functionality

### Production Considerations

- **Database**: Consider upgrading from SQLite to PostgreSQL for production
- **File Storage**: Implement cloud storage (AWS S3, Cloudinary) for resume files
- **Security**: Use proper CORS settings and secure JWT secrets
- **Monitoring**: Add logging and error tracking (Sentry, LogRocket)

### Troubleshooting

- **CORS Issues**: Ensure your backend CORS settings include your frontend domain
- **Database Issues**: Check if the database is properly initialized on first deployment
- **Environment Variables**: Verify all required environment variables are set correctly