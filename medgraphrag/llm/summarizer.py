import time

import tiktoken
from openai import OpenAI

from medgraphrag.llm.config import get_chat_model, openai_client_kwargs


_SUMMARY_PROMPT = """
请用中文总结给定的医学文本或药品说明书。只输出与原文相关的类别，格式为"类别：关键信息"，不要添加额外解释。

可用类别：
药品名称：药品通用名、商品名、活性成分。
适应症：药品用于治疗或预防的疾病/状态。
用法用量：推荐剂量、给药频率、疗程、剂量调整。
给药途径：口服、静脉、玻璃体腔内注射等。
禁忌：禁止使用或不应使用的情况。
警告注意事项：严重风险、监测要求、使用前注意事项。
不良反应：常见或严重不良反应。
药物相互作用：与其他药物或治疗的相互影响。
特殊人群：妊娠、哺乳、儿童、老年、肝肾功能异常等。
检查监测：需要检查或随访的指标。
疾病/症状：文本中涉及的疾病、症状或并发症。
解剖部位：涉及的器官、组织或部位。
"""


def _call_llm(chunk: str, max_retries: int = 5) -> str:
    client = OpenAI(**openai_client_kwargs())
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=get_chat_model(),
                messages=[
                    {"role": "system", "content": _SUMMARY_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as exc:
            message = str(exc)
            is_rate_limit = "429" in message or "速率限制" in message
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            wait_seconds = min(60, 5 * (2 ** attempt))
            print(f"[Summary] 触发速率限制，等待 {wait_seconds} 秒后重试...")
            time.sleep(wait_seconds)


def _split_into_chunks(text: str, tokens: int = 500) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        encoding = tiktoken.encoding_for_model(get_chat_model())
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    words = encoding.encode(text)
    chunks = []
    for i in range(0, len(words), tokens):
        chunk = encoding.decode(words[i:i + tokens]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def process_chunks(content: str) -> str:
    chunks = _split_into_chunks(content)
    responses = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"[Summary] 处理摘要块 {idx}/{len(chunks)}")
        response = _call_llm(chunk)
        if response and response.strip():
            responses.append(response.strip())

    summary = "\n\n".join(responses).strip()
    if not summary:
        print("[Summary] 未生成有效摘要")
    return summary
