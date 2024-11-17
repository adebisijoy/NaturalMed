from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uuid
from plant_rag import PlantRAGSystem

load_dotenv()

required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

app = FastAPI(title="Plant Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

plant_rag_system = PlantRAGSystem()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

class ChatMessage(BaseModel):
    session_id: str = ""
    message: str

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        if not message.session_id:
            message.session_id = str(uuid.uuid4())

        response = plant_rag_system.chat(
            session_id=message.session_id,
            question=message.message
        )

        return {
            "session_id": message.session_id,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_with_image")
async def chat_with_image(
    file: UploadFile = File(...),
    session_id: str = Form(""),
    message: str = Form(...)
):
    try:
        if not session_id:
            session_id = str(uuid.uuid4())

        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
            )
        
        response = plant_rag_system.chat(
            session_id=session_id,
            question=message,
            image_data=contents
        )
        
        return {
            "session_id": session_id,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    try:
        history = plant_rag_system.get_chat_history(session_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_plant_guide")
async def upload_plant_guide(file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDFs allowed."
            )
        
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
            )
        
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
        
        plant_rag_system.process_pdf(temp_path)
        os.remove(temp_path)
        
        return JSONResponse(
            content={"message": "Plant guide processed successfully"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)