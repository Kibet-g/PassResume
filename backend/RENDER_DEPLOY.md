# PassResume Backend - Render Deployment

## Quick Deploy to Render

1. **Create a new Web Service on Render:**
   - Go to [render.com](https://render.com) and sign up/login
   - Click "New +" → "Web Service"
   - Connect your GitHub repository or upload this backend folder

2. **Configuration:**
   - **Name:** `passresume-backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT app_enhanced:app`
   - **Instance Type:** Free tier is sufficient for testing

3. **Environment Variables:**
   - `FLASK_ENV=production`
   - `PYTHONPATH=/opt/render/project/src`

4. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Your backend will be available at: `https://passresume-backend.onrender.com`

## Alternative: Manual Deployment

If you prefer to deploy manually, you can:

1. **Zip the backend folder**
2. **Upload to Render** via their dashboard
3. **Configure** as described above

## Performance Benefits

Render offers:
- ✅ Better Python support than Railway
- ✅ Faster cold starts
- ✅ More reliable database connections
- ✅ Better logging and monitoring
- ✅ Free SSL certificates
- ✅ Automatic deployments from Git

## Next Steps

After deployment:
1. Test the health endpoint: `https://your-app.onrender.com/health`
2. Update frontend environment variable to point to new backend URL
3. Redeploy frontend to Netlify with updated backend URL