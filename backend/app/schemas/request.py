from pydantic import BaseModel, Field

class VerifyRequest(BaseModel):
    """
    Optional Pydantic model for verification if using JSON bodies instead of Form data.
    The primary endpoint uses Form Data (multipart/form-data) to support file uploads.
    """
    text: str = Field(None, description="The claim or text to be verified")
