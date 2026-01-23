from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from recommender_service import rec_service
from config import PROCESSED_DATA_DIR
import uvicorn

app = FastAPI(title="Causal Rating Improver API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserSnapshot(BaseModel):
    current_rating: float
    recent_accuracy: float = 0.5
    recent_avg_difficulty: float = 1200.0

@app.on_event("startup")
def startup_event():
    # Auto-seed data if Parquet files don't exist (for deployment)
    causal_file = PROCESSED_DATA_DIR / "causal_att_effects.parquet"
    if not causal_file.exists():
        print("Data files not found. Running seed_dummy_data...")
        from seed_dummy_data import seed_data
        seed_data()
    rec_service.load_data()

class HandleRequest(BaseModel):
    handle: str

@app.post("/analyze-profile")
async def analyze_profile(req: HandleRequest):
    """Fetches user stats from Local Dataset for recommender."""
    profile = await rec_service.lookup_user_profile(req.handle)
    if not profile:
         raise HTTPException(status_code=404, detail="User not found in local dataset")
    return profile

@app.post("/recommend")
def get_recommendations(user: UserSnapshot):
    try:
        recs = rec_service.recommend(
            user.current_rating, 
            user.recent_accuracy, 
            user.recent_avg_difficulty
        )
        if not recs:
            return {"status": "no_match", "message": "No strong causal signals found for this specific profile. Try solving more diverse problems!"}
            
        return {"status": "success", "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/global-insights")
def get_global_insights():
    """Returns top problems by causal impact."""
    return rec_service.get_global_insights()

@app.get("/problem-details/{problem_id}")
def get_problem_details(problem_id: str):
    """Returns stats, similar users, and next steps for a problem."""
    details = rec_service.get_problem_details(problem_id)
    if not details:
        raise HTTPException(status_code=404, detail="Problem not found in causal registry")
    return details
    
@app.get("/health")
def health():
    return {"status": "active"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
