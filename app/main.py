from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#for docker
from app.prediction import lstm_predict, cnn_predict, predict, predict_csv

#for local
# from prediction import lstm_predict, cnn_predict, predict, predict_csv

class TextInput(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/lstm-predict")
async def lstm_predict_text(text_input: TextInput):
    try:
        text = text_input.text
        response = predict(text, lstm_predict)
        return JSONResponse(content={"response":response})
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/cnn-predict")
async def cnn_predict_text(text_input: TextInput):
    try:
        text = text_input.text
        response = predict(text, cnn_predict)
        return JSONResponse(content={"response":response})
    except Exception as e:
        return {"error": str(e)}

@app.post("/lstm-predict-csv")
async def lstm_predict_text_csv(file: UploadFile = File(...)):
    try:
        file_name = file.filename
        file = await file.read()
        response = predict_csv(file_name, file, lstm_predict)
        return JSONResponse(content={"response":response})
    except Exception as e:
        return {"error": str(e)}

@app.post("/cnn-predict-csv")
async def cnn_predict_text_csv(file: UploadFile = File(...)):
    try:
        file_name = file.filename
        file = await file.read()
        response = predict_csv(file_name, file, cnn_predict)
        return JSONResponse(content={"response":response})
    except Exception as e:
        return {"error": str(e)}