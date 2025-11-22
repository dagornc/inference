# Liquid Glass Chatbot - Scripts de Démarrage

## 📋 Vue d'ensemble

Scripts pour démarrer et arrêter tous les services du chatbot Liquid Glass.

## 🚀 Utilisation

### Démarrage des services

```bash
./start.sh
```

### Options disponibles

```bash
./start.sh --version    # Affiche la version
./start.sh --info       # Affiche les informations détaillées
./start.sh --restart    # Redémarre tous les services
./start.sh --help       # Affiche l'aide
```

### Arrêt des services

```bash
./stop.sh
```

## 🐳 Services démarrés

| Service | Port | URL | Description |
|---------|------|-----|-------------|
| **Ollama** | 11434 | http://localhost:11434 | Serveur LLM local |
| **Open WebUI** | 3000 | http://localhost:3000 | Interface web pour Ollama |
| **ChromaDB** | 8001 | http://localhost:8001 | Base de données vectorielle |
| **Backend API** | 8000 | http://localhost:8000 | FastAPI + RAG Pipeline |
| **Frontend** | 5173 | http://localhost:5173 | React/Vite + Tailwind v4 |

## 📝 Logs

Les logs sont disponibles dans le dossier `logs/` :
- `logs/backend.log` - Logs du backend FastAPI
- `logs/frontend.log` - Logs du frontend Vite

## ⚙️ Configuration

Les fichiers de configuration se trouvent dans `config/` :
- `global.yaml` - Configuration globale
- `01_embedding_v2.yaml` - Configuration de l'embedding
- `02_retrieval_v2.yaml` - Configuration de la récupération
- `03_reranking_v2.yaml` - Configuration du reranking
- `04_compression_v2.yaml` - Configuration de la compression
- `05_generation_v2.yaml` - Configuration de la génération

## 🔧 Prérequis

- Docker installé et démarré
- Node.js et npm installés
- Python 3.9+ avec environnement virtuel `.venv`

## 📊 Vérification du statut

Après le démarrage, le script affiche automatiquement le statut de tous les services.

Pour vérifier manuellement :

```bash
# Vérifier les conteneurs Docker
docker ps

# Vérifier le backend
curl http://localhost:8000/health

# Vérifier le frontend
curl http://localhost:5173
```

## 🛑 Arrêt manuel

Si vous avez besoin d'arrêter les services manuellement :

```bash
# Arrêter les conteneurs Docker
docker stop ollama open-webui chromadb

# Arrêter le backend
pkill -f "uvicorn src.api.main:app"

# Arrêter le frontend
pkill -f "vite"
```

## 🎯 Exemple d'utilisation

```bash
# Première utilisation
./start.sh

# Redémarrage après modification du code
./start.sh --restart

# Afficher les informations
./start.sh --info

# Arrêter tous les services
./stop.sh
```

## 🐛 Dépannage

### Docker n'est pas démarré
```
[ERROR] Docker n'est pas démarré. Veuillez démarrer Docker.
```
**Solution**: Démarrez Docker Desktop

### Port déjà utilisé
Si un port est déjà utilisé, arrêtez le service existant :
```bash
lsof -ti:8000 | xargs kill -9  # Pour le port 8000
```

### Environnement virtuel non trouvé
```
[ERROR] Environnement virtuel .venv non trouvé
```
**Solution**: Créez l'environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

## 📚 Documentation

Pour plus d'informations, consultez :
- [walkthrough.md](walkthrough.md) - Guide complet d'implémentation
- [QUICKSTART.md](QUICKSTART.md) - Guide de démarrage rapide
- http://localhost:8000/docs - Documentation Swagger de l'API
