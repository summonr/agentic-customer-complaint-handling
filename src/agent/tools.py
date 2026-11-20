"""Expose Mailpit operations as LangChain tools."""

from langchain.tools import tool
    
from utilities import mailpit

detection = tool("detection")(mailpit.detect_email)
classify = tool("classify")(mailpit.forward_to_review)
