# -*- coding: utf-8 -*-

from fastapi import HTTPException
from pydantic import BaseModel
from rag_backend.util import logger
from rag_backend.util.api_helper import get_rag_agent

class AskRequest(BaseModel):
    query: str

async def ask_question(request: AskRequest):
    """Answers a question based on the knowledge base."""
    # rag_chain = get_rag_chain()
    # try:
    #     logger.info(f"Received question: {request.query}")
    #     answer = rag_chain.invoke(request.query)
    #     logger.info(f"Answer: {answer}")
    #     return {"answer": answer}
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     logger.error(f"Failed to answer question: {e}")
    #     raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")

    rag_agent = get_rag_agent()
    try:
        logger.info(f"Received question: {request.query}")
        steps = list(rag_agent.stream(
            {"messages": [{"role": "user", "content": request.query}]},
            stream_mode="values",
        ))
        # 本RAG没有记忆功能，只返回AI的回复
        answer = steps[-1]["messages"][-1].content
        logger.info(f"Answer: {answer}")
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")