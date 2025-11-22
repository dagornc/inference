#!/bin/bash

# Liquid Glass Chatbot - Stop Script
# Arrête tous les services

set -e

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

info "Arrêt de tous les services..."
echo ""

# Arrêt des conteneurs Docker
info "Arrêt des conteneurs Docker..."
docker stop ollama open-webui chromadb 2>/dev/null || true
success "Conteneurs Docker arrêtés"

# Arrêt du backend
info "Arrêt du Backend API..."
pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
success "Backend arrêté"

# Arrêt du frontend
info "Arrêt du Frontend..."
pkill -f "vite" 2>/dev/null || true
success "Frontend arrêté"

echo ""
success "Tous les services sont arrêtés !"
