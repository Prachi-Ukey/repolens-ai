import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.file import RepositoryFile

from app.schemas.chat import (
    QuestionRequest,
    QuestionResponse,
    ConversationResponse,
    MessageResponse,
)

from app.services.auth_service import get_current_user
from app.services.repo_service import get_repository_by_id

from app.ai.rag import RAGEngine
from app.ai.vector_store import get_vector_store
from app.ai.llm import get_llm_provider


router = APIRouter(
    prefix="/repositories/{repo_id}",
    tags=["Chat & RAG"]
)


# ============================================================
# RAG ENGINE
# ============================================================

rag_engine = RAGEngine(
    vector_store=get_vector_store(),
    llm_provider=get_llm_provider()
)


# ============================================================
# ASK QUESTION
# ============================================================

@router.post(
    "/chat",
    response_model=QuestionResponse
)
def ask_question(
    repo_id: str,
    request: QuestionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    repo = get_repository_by_id(
        db,
        repo_id,
        current_user.id
    )

    if not repo:

        raise HTTPException(
            status_code=404,
            detail="Repository not found."
        )

    # ---------------------------------------------------------
    # Fetch or create conversation
    # ---------------------------------------------------------

    if request.conversation_id:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == request.conversation_id,
                Conversation.repository_id == repo_id
            )
            .first()
        )

        if not conversation:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found."
            )

    else:

        title = request.question[:40]

        if len(request.question) > 40:
            title += "..."

        conversation = Conversation(
            repository_id=repo_id,
            user_id=current_user.id,
            title=title
        )

        db.add(conversation)

        db.flush()

    # ---------------------------------------------------------
    # Save user message
    # ---------------------------------------------------------

    user_msg = Message(
        conversation_id=conversation.id,
        sender="user",
        content=request.question
    )

    db.add(user_msg)

    # ---------------------------------------------------------
    # Execute RAG
    #
    # Retrieve up to 20 candidate chunks.
    # RAGEngine performs the final relevance filtering.
    # ---------------------------------------------------------

    rag_result = rag_engine.ask_question(
        repo_id=repo_id,
        question=request.question,
        top_k=20
    )

    answer = rag_result.get(
        "answer",
        ""
    )

    sources = rag_result.get(
        "sources",
        []
    )

    grounded = rag_result.get(
        "grounded",
        False
    )

    # ---------------------------------------------------------
    # Save sources
    # ---------------------------------------------------------

    sources_json = json.dumps(
        sources
    )

    # ---------------------------------------------------------
    # Save assistant message
    # ---------------------------------------------------------

    assistant_msg = Message(
        conversation_id=conversation.id,
        sender="assistant",
        content=answer,
        sources_json=sources_json
    )

    db.add(assistant_msg)

    db.commit()

    # ---------------------------------------------------------
    # Return response
    # ---------------------------------------------------------

    return QuestionResponse(
        conversation_id=conversation.id,
        question=request.question,
        answer=answer,
        sources=sources,
        grounded=grounded
    )


# ============================================================
# GET CONVERSATIONS
# ============================================================

@router.get(
    "/conversations",
    response_model=List[ConversationResponse]
)
def get_conversations(
    repo_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    repo = get_repository_by_id(
        db,
        repo_id,
        current_user.id
    )

    if not repo:

        raise HTTPException(
            status_code=404,
            detail="Repository not found."
        )

    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.repository_id == repo_id,
            Conversation.user_id == current_user.id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .all()
    )

    result = []

    for conv in conversations:

        msgs = (
            db.query(Message)
            .filter(
                Message.conversation_id == conv.id
            )
            .order_by(
                Message.created_at.asc()
            )
            .all()
        )

        conv_resp = ConversationResponse(
            id=conv.id,
            repository_id=conv.repository_id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=[
                MessageResponse.model_validate(
                    message
                )
                for message in msgs
            ]
        )

        result.append(
            conv_resp
        )

    return result


# ============================================================
# SEARCH REPOSITORY
# ============================================================

@router.post("/search")
def search_repository(
    repo_id: str,
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    query = payload.get(
        "query",
        ""
    ).strip()

    search_type = payload.get(
        "type",
        "semantic"
    )

    repo = get_repository_by_id(
        db,
        repo_id,
        current_user.id
    )

    if not repo:

        raise HTTPException(
            status_code=404,
            detail="Repository not found."
        )

    if not query:

        return {
            "results": []
        }

    results = []

    # ---------------------------------------------------------
    # Keyword search
    # ---------------------------------------------------------

    if search_type == "keyword":

        files = (
            db.query(RepositoryFile)
            .filter(
                RepositoryFile.repository_id == repo_id,
                (
                    RepositoryFile.file_path.ilike(
                        f"%{query}%"
                    )
                    |
                    RepositoryFile.content.ilike(
                        f"%{query}%"
                    )
                )
            )
            .limit(20)
            .all()
        )

        for file in files:

            results.append(
                {
                    "file_path": file.file_path,
                    "file_id": file.id,
                    "language": file.language,
                    "match_type": "keyword",
                    "snippet": (
                        file.content[:200]
                        if file.content
                        else ""
                    )
                }
            )

    # ---------------------------------------------------------
    # Semantic search
    # ---------------------------------------------------------

    else:

        chunks = (
            rag_engine.vector_store
            .query_similar_chunks(
                repo_id,
                query,
                top_k=10
            )
        )

        for chunk in chunks:

            results.append(
                {
                    "file_path": chunk["file_path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "symbol_name": chunk["symbol_name"],
                    "match_type": "semantic",
                    "snippet": chunk["content"]
                }
            )

    return {
        "query": query,
        "type": search_type,
        "results": results
    }