import os

from dotenv import load_dotenv


load_dotenv(override=True)


DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_GLM_CHAT_MODEL = "glm-4.5"
DEFAULT_GLM_EMBEDDING_MODEL = "embedding-3"


def _using_gpt() -> bool:
    return bool(os.getenv("GPT_API_KEY"))


def _using_glm() -> bool:
    return bool(os.getenv("GLM_API_KEY") or os.getenv("GLM_API_BASE_URL"))


def get_api_key() -> str | None:
    return (
        os.getenv("GPT_API_KEY")
        or os.getenv("GLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def get_base_url() -> str | None:
    if os.getenv("GPT_API_BASE_URL"):
        return os.getenv("GPT_API_BASE_URL")
    if os.getenv("GLM_API_BASE_URL"):
        return os.getenv("GLM_API_BASE_URL")
    if os.getenv("OPENAI_API_BASE_URL"):
        return os.getenv("OPENAI_API_BASE_URL")
    if os.getenv("GLM_API_KEY"):
        return DEFAULT_GLM_BASE_URL
    return None


def get_chat_model(openai_default: str = "gpt-4-1106-preview") -> str:
    if _using_gpt():
        return os.getenv("GPT_CHAT_MODEL") or openai_default
    if _using_glm():
        return (
            os.getenv("GLM_CHAT_MODEL")
            or os.getenv("LLM_CHAT_MODEL")
            or DEFAULT_GLM_CHAT_MODEL
        )
    return os.getenv("OPENAI_CHAT_MODEL") or os.getenv("LLM_CHAT_MODEL") or openai_default


def openai_client_kwargs() -> dict:
    kwargs = {"api_key": get_api_key()}
    base_url = get_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs
