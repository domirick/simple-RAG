from langchain.docstore.document import Document

def pretty_print_docs(docs:list[Document]) -> str:
    pretty_str = f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )

    return pretty_str

def pretty_print_docs_with_sources(docs:list[Document]) -> str:
    pretty_str = f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content + "\n\nSource:" + d.metadata["source"] for i, d in enumerate(docs)]
        )

    return pretty_str