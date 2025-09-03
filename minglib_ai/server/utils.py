

import uuid
import time
import re
from datetime import datetime

def parse_response(response: str, start_time,  model_type: str = "no_rag") -> dict:
    """Parse a raw LLM response into the expected API structure"""

    try:
        # Try to extract a reference if mentioned, otherwise mark as no reference
        reference_match = re.search(r'REFERENCE:\s*([^\n]+)', response, re.IGNORECASE)
        reference = reference_match.group(1).strip() if reference_match else "No RAG - General Knowledge"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "data": {
                "id": f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
                "content": response,
                "reference": reference,
                "model_used": model_type,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": str(e),
                "code": "PARSE_ERROR",
                "details": {"model_type": model_type, "response": response}
            }
        }