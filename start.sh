#!/bin/bash

# Liquid Glass Chatbot - Startup Script
# Démarre tous les services nécessaires : Ollama, Open WebUI, ChromaDB, Backend, Frontend

set -e

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'affichage des informations
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Fonction d'affichage de la version
show_version() {
    echo "Liquid Glass Chatbot - Version ${VERSION}"
    echo "Backend: FastAPI + RAG Pipeline"
    echo "Frontend: React/Vite + Tailwind CSS v4"
    echo "Services: Ollama, Open WebUI, ChromaDB"
}

# Fonction d'affichage de l'aide
show_help() {
    cat << EOF
Usage: ./start.sh [OPTIONS]

Options:
    --restart       Arrête et redémarre tous les services
    --version       Affiche la version du script
    --info          Affiche les informations détaillées sur les services
    -h, --help      Affiche cette aide

Services démarrés:
    - Ollama (LLM local)
    - Open WebUI (Interface web pour Ollama)
    - ChromaDB (Base de données vectorielle)
    - Backend API (FastAPI - Port 8000)
    - Frontend (React/Vite - Port 5173)

Exemples:
    ./start.sh              # Démarre tous les services
    ./start.sh --restart    # Redémarre tous les services
    ./start.sh --info       # Affiche les informations
EOF
}

# Fonction d'affichage des informations détaillées
show_info() {
    cat << EOF
${BLUE}=== Liquid Glass Chatbot - Informations ===${NC}

${GREEN}Services:${NC}
  1. Ollama
     - Port: 11434
     - Description: Serveur LLM local
     - Modèle par défaut: llama3
     - Docker: ollama/ollama

  2. Open WebUI
     - Port: 3000
     - Description: Interface web pour Ollama
     - URL: http://localhost:3000
     - Docker: ghcr.io/open-webui/open-webui

  3. ChromaDB
     - Port: 8001
     - Description: Base de données vectorielle
     - URL: http://localhost:8001
     - Docker: chromadb/chroma

  4. Backend API
     - Port: 8000
     - Description: FastAPI + RAG Pipeline (5 étapes)
     - URL: http://localhost:8000
     - Docs: http://localhost:8000/docs

  5. Frontend
     - Port: 5173
     - Description: React/Vite + Tailwind CSS v4
     - URL: http://localhost:5173

${GREEN}Architecture RAG Pipeline:${NC}
  Step 1: Query Expansion (Embedding)
  Step 2: Hybrid Retrieval (Dense + Sparse)
  Step 3: Multi-Stage Reranking
  Step 4: Contextual Compression
  Step 5: Advanced Generation

${GREEN}Fichiers de configuration:${NC}
  - config/global.yaml
  - config/01_embedding_v2.yaml
  - config/02_retrieval_v2.yaml
  - config/03_reranking_v2.yaml
  - config/04_compression_v2.yaml
  - config/05_generation_v2.yaml

${GREEN}Logs:${NC}
  - Backend: logs/backend.log
  - Frontend: logs/frontend.log
EOF
}

# Fonction pour arrêter les services
stop_services() {
    info "Arrêt des services..."
    
    # Arrêt des conteneurs Docker
    docker stop ollama open-webui chromadb 2>/dev/null || true
    
    # Arrêt du backend
    pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
    
    # Arrêt du frontend
    pkill -f "vite" 2>/dev/null || true
    
    success "Services arrêtés"
}

# Fonction pour vérifier Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé. Veuillez installer Docker."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker n'est pas démarré. Veuillez démarrer Docker."
        exit 1
    fi
    
    success "Docker est disponible"
}

# Fonction pour démarrer Ollama
start_ollama() {
    info "Démarrage d'Ollama..."
    
    if docker ps | grep -q ollama; then
        warning "Ollama est déjà en cours d'exécution"
    else
        docker run -d --rm \
            --name ollama \
            -p 11434:11434 \
            -v ollama:/root/.ollama \
            ollama/ollama
        
        sleep 5
        
        # Vérification et pull du modèle llama3
        info "Vérification du modèle llama3..."
        docker exec ollama ollama pull llama3 2>/dev/null || true
        
        success "Ollama démarré (Port 11434)"
    fi
}

# Fonction pour démarrer Open WebUI
start_openwebui() {
    info "Démarrage d'Open WebUI..."
    
    if docker ps | grep -q open-webui; then
        warning "Open WebUI est déjà en cours d'exécution"
    else
        docker run -d --rm \
            --name open-webui \
            -p 3000:8080 \
            -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
            -v open-webui:/app/backend/data \
            ghcr.io/open-webui/open-webui:main
        
        success "Open WebUI démarré (Port 3000)"
    fi
}

# Fonction pour démarrer ChromaDB
start_chromadb() {
    info "Démarrage de ChromaDB..."
    
    if docker ps | grep -q chromadb; then
        warning "ChromaDB est déjà en cours d'exécution"
    else
        docker run -d --rm \
            --name chromadb \
            -p 8001:8000 \
            -v chromadb:/chroma/chroma \
            chromadb/chroma
        
        success "ChromaDB démarré (Port 8001)"
    fi
}

# Fonction pour démarrer le backend
start_backend() {
    info "Démarrage du Backend API..."
    
    cd "${SCRIPT_DIR}"
    
    # Création du dossier logs
    mkdir -p logs
    
    # Activation de l'environnement virtuel
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        error "Environnement virtuel .venv non trouvé"
        exit 1
    fi
    
    # Démarrage du backend en arrière-plan
    nohup python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 \
        > logs/backend.log 2>&1 &
    
    sleep 2
    success "Backend API démarré (Port 8000)"
}

# Fonction pour démarrer le frontend
start_frontend() {
    info "Démarrage du Frontend..."
    
    cd "${SCRIPT_DIR}/frontend"
    
    # Création du dossier logs
    mkdir -p ../logs
    
    # Démarrage du frontend en arrière-plan
    nohup npm run dev > ../logs/frontend.log 2>&1 &
    
    sleep 2
    success "Frontend démarré (Port 5173)"
}

# Fonction pour afficher le statut des services
show_status() {
    echo ""
    info "Statut des services:"
    echo ""
    
    # Ollama
    if docker ps | grep -q ollama; then
        echo -e "  ${GREEN}✓${NC} Ollama          http://localhost:11434"
    else
        echo -e "  ${RED}✗${NC} Ollama          (non démarré)"
    fi
    
    # Open WebUI
    if docker ps | grep -q open-webui; then
        echo -e "  ${GREEN}✓${NC} Open WebUI      http://localhost:3000"
    else
        echo -e "  ${RED}✗${NC} Open WebUI      (non démarré)"
    fi
    
    # ChromaDB
    if docker ps | grep -q chromadb; then
        echo -e "  ${GREEN}✓${NC} ChromaDB        http://localhost:8001"
    else
        echo -e "  ${RED}✗${NC} ChromaDB        (non démarré)"
    fi
    
    # Backend
    if pgrep -f "uvicorn src.api.main:app" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Backend API     http://localhost:8000"
        echo -e "                      http://localhost:8000/docs (Swagger)"
    else
        echo -e "  ${RED}✗${NC} Backend API     (non démarré)"
    fi
    
    # Frontend
    if pgrep -f "vite" > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Frontend        http://localhost:5173"
    else
        echo -e "  ${RED}✗${NC} Frontend        (non démarré)"
    fi
    
    echo ""
    info "Logs disponibles dans le dossier: ${SCRIPT_DIR}/logs/"
    echo ""
}

# Fonction principale
main() {
    # Gestion des arguments
    case "${1:-}" in
        --version)
            show_version
            exit 0
            ;;
        --info)
            show_info
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --restart)
            info "Redémarrage de tous les services..."
            stop_services
            sleep 2
            ;;
        "")
            # Pas d'argument, démarrage normal
            ;;
        *)
            error "Option inconnue: $1"
            show_help
            exit 1
            ;;
    esac
    
    # Démarrage des services via Docker Compose (si le fichier docker-compose.yml existe)
    if [ -f "${SCRIPT_DIR}/docker-compose.yml" ]; then
        info "Lancement des services Docker via docker-compose..."
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d
    else
        warning "docker-compose.yml non trouvé, utilisation des commandes docker run individuelles."
    fi}

# Exécution du script
main "$@"
