from pathlib import Path


def _load_pdf(datapath):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "Reading PDF files requires pypdf. Install it with `pip install pypdf` "
            "or recreate the conda environment from medgraphrag.yml."
        ) from exc

    reader = PdfReader(datapath)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_high(datapath: str) -> str:
    path = Path(datapath)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(datapath)

    all_content = ""
    with open(datapath, "r", encoding="utf-8") as file:
        for line in file:
            all_content += line.strip() + "\n"
    return all_content
