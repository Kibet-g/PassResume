# Test your new Render backend
# Replace YOUR-APP-NAME with your actual Render app name

# Test health endpoint
curl https://YOUR-APP-NAME.onrender.com/health

# Expected response:
# {"database_connected":true,"ml_model_loaded":true,"status":"healthy","timestamp":"..."}

# If you get this response, your backend is working perfectly!