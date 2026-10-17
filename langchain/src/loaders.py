from langchain_community.document_loaders import PyPDFLoader

path1 = "knowledge/expense_policy.pdf"
path2 = "knowledge/employee_handbook.pdf"


expensePolicyPages = PyPDFLoader(path1).load()
employeeHandbookPages = PyPDFLoader(path2).load()

print(f"Pages loaded: {len(expensePolicyPages)}")
print(f"Pages loaded: {len(employeeHandbookPages)}")