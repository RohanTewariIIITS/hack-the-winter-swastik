# Deployment Guide

## Step 1: Deploy Backend to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project (from backend root)
cd /path/to/causal_rating_improver
railway init

# Deploy
railway up

# Get your backend URL
railway open
# Copy the URL (e.g., https://causal-rating-improver.railway.app)
```

## Step 2: Deploy Frontend to Vercel

```bash
# Go to frontend directory
cd frontend

# Create .env file with your Railway backend URL
echo "VITE_API_URL=https://YOUR-RAILWAY-URL.railway.app" > .env

# Build the app
npm run build

# Deploy to Vercel
npx vercel --prod
```

## Step 3: Configure CORS (Important!)

Update `main.py` to allow your Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://your-app.vercel.app"  # Add your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Alternative: Render.com

1. Go to https://render.com
2. Connect your GitHub repo
3. Create a "Web Service" for backend (Python)
4. Create a "Static Site" for frontend

## Files Created for Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies for Railway/Render |
| `Procfile` | Start command for backend |
| `frontend/vercel.json` | Vercel SPA routing config |
| `frontend/.env.example` | Environment variable template |

## Environment Variables

### Backend (Railway)
- No special env vars needed (uses local Parquet files)

### Frontend (Vercel)
- `VITE_API_URL` - Your Railway backend URL
