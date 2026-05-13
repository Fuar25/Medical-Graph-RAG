import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from medgraphrag.llm.client import openai_complete_if_cache


class _FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str = "fake error"):
        super().__init__(message)
        self.status_code = status_code


class _FakeCompletions:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    async def create(self, **kwargs):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=outcome)),
            ]
        )


class _FakeAsyncOpenAI:
    outcomes = []
    instances = []

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(self.__class__.outcomes)
        )
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class OpenAIClientRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_recoverable_error_then_returns_content(self):
        _FakeAsyncOpenAI.outcomes = [
            _FakeStatusError(500, "temporary server failure"),
            "ok content",
        ]
        _FakeAsyncOpenAI.instances = []

        with (
            patch("medgraphrag.llm.client.AsyncOpenAI", _FakeAsyncOpenAI),
            patch("medgraphrag.llm.client.openai_client_kwargs", return_value={}),
            patch("medgraphrag.llm.client.asyncio.sleep", new=AsyncMock()),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                result = await openai_complete_if_cache("test-model", "hello")

        self.assertEqual(result, "ok content")
        self.assertEqual(len(_FakeAsyncOpenAI.instances), 2)

    async def test_does_not_retry_non_recoverable_error(self):
        _FakeAsyncOpenAI.outcomes = [
            _FakeStatusError(401, "invalid api key"),
            "should not be used",
        ]
        _FakeAsyncOpenAI.instances = []

        with (
            patch("medgraphrag.llm.client.AsyncOpenAI", _FakeAsyncOpenAI),
            patch("medgraphrag.llm.client.openai_client_kwargs", return_value={}),
            patch("medgraphrag.llm.client.asyncio.sleep", new=AsyncMock()),
        ):
            with self.assertRaises(_FakeStatusError):
                await openai_complete_if_cache("test-model", "hello")

        self.assertEqual(len(_FakeAsyncOpenAI.instances), 1)


if __name__ == "__main__":
    unittest.main()
