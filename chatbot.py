from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Chargement de la base vectorielle...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="llama3.2", temperature=0.1)

prompt_template = PromptTemplate.from_template("""Tu es EduBot, un assistant pour etudiants en reseaux.
Reponds en FRANCAIS uniquement a partir du contexte fourni.
Si la reponse n'est pas dans le contexte, dis :
"Je ne trouve pas cette information dans les cours."

Contexte : {context}
Question : {question}
Reponse :""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

print("EduBot pret ! Tape quit pour quitter.")
print("-" * 50)

while True:
    question = input("\nTa question : ")
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question.strip():
        continue
    docs = retriever.invoke(question)
    reponse = chain.invoke(question)
    print("\nEduBot :", reponse)
    print("\nSources :")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        print(f"  [{i}] {src} - page {page}")
    print("-" * 50)