#!/bin/bash

# Liquid Glass Chatbot - Pipelines Startup Script
# Démarre le service Open WebUI Pipelines

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

info "Démarrage du service Open WebUI Pipelines..."

# Vérifier si Docker est lancé
if ! docker info > /dev/null 2>&1; then
    echo "Erreur: Docker n'est pas lancé."
    exit 1
fi

# Aller dans le dossier pipelines
cd "${SCRIPT_DIR}/pipelines"

# Lancer le service
docker compose up -d

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
