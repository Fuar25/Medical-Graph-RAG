from typing import List

from medgraphrag.llm.config import get_api_key, get_base_url, get_chat_model


def _get_propositions(text, runnable, extraction_chain) -> list[str]:
    runnable_output = runnable.invoke({"input": text}).content
    return extraction_chain.run(runnable_output)[0].sentences


def run_chunk(essay: str) -> list[str]:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import create_extraction_chain_pydantic
    from langchain_core.pydantic_v1 import BaseModel
    from langchain import hub
    from medgraphrag.ingestion.chunking.agentic import AgenticChunker

    class Sentences(BaseModel):
        sentences: List[str]

    obj = hub.pull("wfh/proposal-indexing")
    llm = ChatOpenAI(
        model=get_chat_model(),
        openai_api_key=get_api_key(),
        openai_api_base=get_base_url(),
    )

    runnable = obj | llm
    extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)

    essay_propositions = []
    for i, para in enumerate(essay.split("\n\n")):
        propositions = _get_propositions(para, runnable, extraction_chain)
        essay_propositions.extend(propositions)
        print(f"Done with {i}")

    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    return ac.get_chunks(get_type='list_of_strings')
