from langchain_text_splitters import RecursiveCharacterTextSplitter
from loaders import expensePolicyPages, employeeHandbookPages

# initialised the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=120,
    add_start_index=True,
)

policyChunks = splitter.split_documents(expensePolicyPages)
employeeChunks = splitter.split_documents(employeeHandbookPages)

print(f"Split {len(expensePolicyPages)} documents into {len(policyChunks)} chunks")
print(f"Split {len(employeeHandbookPages)} documents into {len(employeeChunks)} chunks")

# print(f"policyChunks content: {policyChunks[0].page_content}")
# print(f"policyChunks metadata: {policyChunks[0].page_content}")