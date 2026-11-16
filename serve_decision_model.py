import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Jev Decision Model Serving Endpoint")

# In production / Kaggle, load model & tokenizer here:
# model, tokenizer = load_model("jev_decision_model")

class QuestionRequest(BaseModel):
    id: str
    type: str
    instructions: str
    options: Dict[str, str]

class DecisionRequest(BaseModel):
    state: str
    questions: List[QuestionRequest]

@app.post("/predict")
def predict_decision(req: DecisionRequest):
    try:
        results = []
        for q in req.questions:
            # Mocking probability distribution output for serving endpoint validation
            options = list(q.options.keys())
            probs = torch.softmax(torch.randn(len(options)), dim=-1).tolist()
            prob_dict = {opt: prob for opt, prob in zip(options, probs)}
            
            results.append({
                "question_id": q.id,
                "probabilities": prob_dict
            })
            
        return {"state": req.state, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting serving endpoint on port 8000...")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Serving script compiled successfully.")
