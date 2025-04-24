from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
import os

from src.models.multinomial import Multinomial
from src.pipelines.simple import SimplePipeline

APP_PORT = os.getenv("APP_PORT", 8000)
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info").upper()

app = FastAPI()

cfg = OmegaConf.load("config.yaml")
model = Multinomial(cfg.model) # TODO: replace with the actual selected model
model_path = to_absolute_path(f"models/{cfg.model.name}")
model = model.load(model_path)

pipeline = SimplePipeline(cfg.pipeline) # TODO: replace with the actual selected model
_, _, X_test, y_test = pipeline.load()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>TikTok Recommender</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h1 class="mb-4">Item Recommender</h1>
            <form method="post" action="/recommend">
                <div class="mb-3">
                    <label for="user_id" class="form-label">Enter User ID</label>
                    <input type="text" class="form-control" name="user_id" required>
                </div>
                <button type="submit" class="btn btn-primary">Recommend</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(user_id: str = Form(...)):
    try:
        recommendations = model.recommend(user_id, top_k=5)
    except Exception as e:
        return HTMLResponse(f"<h2>Error: {str(e)}</h2>")

    item_cards = ""
    for item_id in recommendations:
        meta_row = X_test[X_test["id"] == item_id]
        if meta_row.empty:
            continue

        row_dict = meta_row.iloc[0].to_dict()

        table_rows = "".join(
            f"<tr><th>{key}</th><td>{value}</td></tr>"
            for key, value in row_dict.items()
        )

        item_cards += f"""
        <div class="card m-2" style="width: 22rem;">
            <div class="card-body">
                <h5 class="card-title">{row_dict.get('title', item_id)}</h5>
                <table class="table table-sm table-bordered table-striped">
                    {table_rows}
                </table>
            </div>
        </div>
        """

    return f"""
    <html>
    <head>
        <title>Recommendations</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h1 class="mb-4">Recommendations for User <code>{user_id}</code></h1>
            <div class="d-flex flex-wrap justify-content-start">
                {item_cards}
            </div>
            <a href="/" class="btn btn-secondary mt-4">Back</a>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=APP_PORT, log_level=APP_LOG_LEVEL)
