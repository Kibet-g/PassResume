🚀 RENDER DEPLOYMENT CHECKLIST

□ 1. Go to render.com
□ 2. Sign up/Login with GitHub
□ 3. Click "New +" → "Web Service"
□ 4. Connect repository: Kibet-g/PassResume
□ 5. Configure service:
   - Name: passresume-backend
   - Environment: Python 3
   - Root Directory: backend
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn --bind 0.0.0.0:$PORT app_enhanced:app
□ 6. Add Environment Variables:
   - FLASK_ENV=production
   - PYTHONPATH=/opt/render/project/src
   - PYTHONUNBUFFERED=1
□ 7. Click "Create Web Service"
□ 8. Wait for deployment (2-5 minutes)
□ 9. Test health endpoint: https://YOUR-APP.onrender.com/health
□ 10. Update frontend with new backend URL

Your backend will be available at:
https://passresume-backend.onrender.com

After deployment, run:
cd frontend
netlify env:set NEXT_PUBLIC_API_URL https://passresume-backend.onrender.com --force
netlify deploy --prod