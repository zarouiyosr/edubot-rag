# EduBot RAG 🎓

Assistant intelligent qui répond aux questions des étudiants
sur leurs cours de réseaux informatiques en français.

## Technologies utilisées
- Python 3.11
- LangChain
- FAISS (base vectorielle)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Ollama + Llama 3.2 (LLM local)

## Installation

### 1. Cloner le projet
git clone https://github.com/TON_USERNAME/edubot-rag.git
cd edubot-rag

### 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate

### 3. Installer les bibliothèques
pip install -r requirements.txt

### 4. Installer Ollama + Llama 3.2
Télécharge Ollama sur ollama.com
ollama pull llama3.2

### 5. Ajouter tes PDFs
Mets tes cours de réseaux dans le dossier documents/

## Utilisation

### Étape 1 — Indexer les documents (une seule fois)
python indexer.py

### Étape 2 — Lancer le chatbot
Terminal 1 : ollama serve
Terminal 2 : python chatbot.py

## Exemple
Ta question : C'est quoi le modèle OSI ?

EduBot : Le modèle OSI est un modèle de référence qui divise
         les communications réseau en 7 couches distinctes...

Sources :
  [1] documents/cours_reseaux_ch1.pdf - page 4

## Auteurs
- Yosser Zaroui
- Wiem Ben ouhiba Dit Ben Aaicha