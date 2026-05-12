from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ETAPE 1 : Chargement des PDFs
print("Chargement des documents...")
loader = DirectoryLoader(
    "documents/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
print(f"  {len(documents)} pages chargees")

# ETAPE 2 : Chunking
print("Decoupage en chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"  {len(chunks)} chunks crees")

# ETAPE 3 : Embeddings
print("Creation des embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ETAPE 4 : Stockage FAISS
print("Stockage dans FAISS...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore/faiss_index")
print("Base vectorielle prete !")