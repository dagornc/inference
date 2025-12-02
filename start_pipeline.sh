#!/bin/bash

# Liquid Glass Chatbot - Pipelines Startup Script
# Démarre le service Open WebUI Pipelines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINES_DIR="${SCRIPT_DIR}/pipelines"
DOCKER_COMPOSE_FILE="${PIPELINES_DIR}/docker-compose.yml"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

stop_pipelines() {
    info "Arrêt des services pipelines existants..."
    cd "${PIPELINES_DIR}" || { error "Impossible de naviguer vers ${PIPELINES_DIR}"; exit 1; }
    if [ -f "${DOCKER_COMPOSE_FILE}" ]; then
        docker compose down --remove-orphans
        success "Services pipelines arrêtés et supprimés."
    else
        info "Aucun fichier docker-compose.yml trouvé, pas de services à arrêter."
    fi
}

info "Démarrage du service Open WebUI Pipelines..."

# Vérifier si Docker est lancé
if ! docker info > /dev/null 2>&1; then
    error "Docker n'est pas lancé. Veuillez démarrer Docker Desktop ou le service Docker."
    exit 1
fi

# Vérifier l'existence du fichier docker-compose.yml
if [ ! -f "${DOCKER_COMPOSE_FILE}" ]; then
    error "Le fichier docker-compose.yml est introuvable à l'emplacement : ${DOCKER_COMPOSE_FILE}"
    exit 1
fi

# Demander à l'utilisateur s'il veut arrêter les pipelines existants
read -p "$(echo -e "${BLUE}[QUESTION]${NC} Voulez-vous arrêter et supprimer les pipelines existants avant de démarrer ? (y/N): ")" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    stop_pipelines
fi

# Aller dans le dossier pipelines
cd "${PIPELINES_DIR}" || { error "Impossible de naviguer vers ${PIPELINES_DIR}"; exit 1; }

# Lancer le service
info "Lancement des services Docker Compose..."
if ! docker compose up -d; then
    error "Échec du démarrage des services Docker Compose. Veuillez vérifier les logs pour plus de détails."
    exit 1
fi

echo ""
success "Service Pipelines démarré avec succès !"
echo ""
echo -e "URL de connexion : ${GREEN}http://host.docker.internal:9099${NC}"
echo -e "Clé API          : ${GREEN}0p3n-w3bu!${NC}"
echo ""
echo "Configuration dans Open WebUI :"
echo "1. Allez dans Admin Panel > Settings > Connections"
echo "2. Ajoutez une connexion OpenAI"
echo "3. Entrez l'URL et la Clé ci-dessus"
echo "4. Activez le pipeline et sélectionnez 'Liquid Glass RAG Pipeline' dans le chat"
