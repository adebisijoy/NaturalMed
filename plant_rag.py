from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import base64
import os
import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from openai import OpenAI

class PlantRAGSystem:
    def __init__(self, index_name: str = "naturalmed"):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.client = OpenAI()

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )

    def _analyze_image(self, image_data: bytes) -> str:
        """Analyze image using Vision API to identify plant."""
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a plant identification expert. Please identify the plant in the image and provide its scientific name and common name. Return ONLY the identified plant name without any additional explanation."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")

    def _get_relevant_context(self, query: str, plant_name: Optional[str] = None) -> str:
        """
        Get relevant context from the vector store based on query and plant name.
        If plant_name is provided, it will be used to enhance the search.
        """
        if plant_name:
            # Create an enhanced query that combines the user's question with the plant name
            enhanced_query = f"{plant_name}"
            k_value = 25 # Get more results since we have more specific context
        else:
            enhanced_query = query
            k_value = 3

        results = self.vector_store.similarity_search(enhanced_query, k=k_value)

        context = ""
        
        for doc in results:
            content = doc.page_content
            local_names = self.extract_local_names(content)
            medicinal_info = self.extract_medicinal_benefits(content)
            
            context += f"{content}\nLocal Names: {local_names}\nMedicinal Benefits: {medicinal_info}\n\n"
        
        return context

    def chat(self, session_id: str, question: str, image_data: Optional[bytes] = None) -> str:
        chat_history = self.get_chat_history(session_id)
        
        plant_name = None
        if image_data:
            # First identify the plant from the image
            plant_name = self._analyze_image(image_data)
        
        # Get context using both the question and plant name (if available)
        context = self._get_relevant_context(question, plant_name)
        
        if image_data:
            response = self._process_with_image(image_data, question, context, chat_history, plant_name)
        else:
            response = self._process_text_only(question, context, chat_history)
        
        self.store_chat(session_id, "user", question)
        self.store_chat(session_id, "assistant", response)
        
        return response

    def _process_with_image(self, image_data: bytes, question: str, context: str, 
                          chat_history: List[Dict[str, Any]], plant_name: str) -> str:
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a plant expert analyzing images and providing detailed information.
                Identified Plant: {plant_name}
                
                Context from Database:
                {context}
                
                Previous Chat:
                {self._format_chat_history(chat_history)}
                
                Focus on:
                1. Confirming or correcting the plant identification
                2. Growing conditions and care requirements
                3. Medicinal properties and traditional uses
                4. Safety considerations and contraindications
                5. Local names and cultural significance"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def _process_text_only(self, question: str, context: str, chat_history: List[Dict[str, Any]]) -> str:
        messages = [
            {
                "role": "system",
                "content": f"""You are a plant expert providing detailed information.
                
                Context from Database:
                {context}
                
                Previous Chat:
                {self._format_chat_history(chat_history)}"""
            },
            {
                "role": "user",
                "content": question
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def extract_local_names(self, content: str) -> str:
        local_names_pattern = r"Local Names\s*:\s*(.*?)(?=\n\n|\Z)"
        match = re.search(local_names_pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return "Local names not found."

    def extract_medicinal_benefits(self, content: str) -> str:
        medicinal_benefits_pattern = r"Medicinal Properties and Traditional Uses\s*:\s*(.*?)(?=\n\n|\Z)"
        match = re.search(medicinal_benefits_pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return "Medicinal benefits not found."

    def store_chat(self, session_id: str, role: str, content: str) -> None:
        chat_entry = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "content": content
        }
        
        embedding = self.embeddings.embed_query(content)
        self.vector_store.add_texts(
            texts=[json.dumps(chat_entry)],
            metadatas=[chat_entry],
            embeddings=[embedding]
        )

    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        results = self.vector_store.similarity_search(
            f"session_id: {session_id}",
            k=10,
            filter={"session_id": session_id}
        )
        
        chat_history = []
        for result in results:
            try:
                entry = json.loads(result.page_content)
                chat_history.append({
                    "role": entry["role"],
                    "content": entry["content"],
                    "timestamp": entry["timestamp"]
                })
            except json.JSONDecodeError:
                continue
        
        return sorted(chat_history, key=lambda x: x["timestamp"])

    def _format_chat_history(self, history: List[Dict[str, Any]]) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in history[-5:]
        ])

    def process_pdf(self, file_path: str) -> None:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(pages)
        
        self.vector_store.add_documents(texts)