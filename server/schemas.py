from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=100, ge=1, le=512)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace")
        return value


class GenerateResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    inference_time_seconds: float
    tokens_per_second: float
