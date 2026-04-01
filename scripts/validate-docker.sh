#!/bin/bash
# scripts/validate-docker.sh - Validação completa em Docker isolado
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERR]${NC} $1"; }

check_prerequisites() {
    log_info "Verificando pré-requisitos..."
    command -v docker >/dev/null 2>&1 || { log_error "Docker não encontrado"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1 || { log_error "Docker Compose não encontrado"; exit 1; }
    for f in Dockerfile.arc-agi3 docker-compose.arc-validation.yml requirements.txt; do
        [[ -f "$f" ]] || { log_error "Arquivo não encontrado: $f"; exit 1; }
    done
    log_success "Pré-requisitos OK"
}

build_image() {
    log_info "Construindo imagem Docker..."
    docker compose -f docker-compose.arc-validation.yml build arc-validator && log_success "Imagem construída" || { log_error "Falha no build"; return 1; }
}

run_validation() {
    local episodes=${1:-10}
    log_info "Validando com $episodes episódios..."
    mkdir -p logs checkpoints submission
    docker compose -f docker-compose.arc-validation.yml run --rm -e VALIDATION_EPISODES=$episodes arc-validator --validate --episodes $episodes --output /app/logs/validation_report.json && log_success "Validação concluída" || { log_error "Validação falhou"; return 1; }
}

build_submission() {
    local version=${1:-v6.3-complete}
    log_info "Construindo submission $version..."
    mkdir -p submission
    docker compose -f docker-compose.arc-validation.yml run --rm -e BUILD_VERSION=$version arc-builder && log_success "Submission criado" || { log_error "Falha no submission"; return 1; }
}

run_tests() {
    log_info "Executando testes..."
    docker compose -f docker-compose.arc-validation.yml run --rm arc-tester && log_success "Testes passaram" || { log_error "Testes falharam"; return 1; }
}

clean() {
    log_warning "Limpando..."
    docker compose -f docker-compose.arc-validation.yml down -v --remove-orphans 2>/dev/null || true
    docker rmi -f arc-agi3-validator:latest arc-agi3-builder:latest 2>/dev/null || true
    log_success "Limpeza concluída"
}

show_help() {
    cat << EOF
🚀 ARC-AGI-3 V6.3 - Validação Docker
Uso: $0 <comando> [opções]
Comandos:
  build           Construir imagem
  validate [n]    Validar com n episódios (default: 10)
  test            Executar testes
  submission [v]  Build submission (versão v)
  all             Build + validate + submission
  clean           Limpar containers
  help            Esta ajuda
Exemplo: $0 all 10
EOF
}

main() {
    local cmd=${1:-help}; shift || true
    check_prerequisites
    case "$cmd" in
        build) build_image ;;
        validate) build_image && run_validation "${1:-10}" ;;
        test) build_image && run_tests ;;
        submission) build_image && build_submission "${1:-v6.3-complete}" ;;
        all) build_image && run_validation "${1:-10}" && build_submission "${2:-v6.3-complete}" ;;
        clean) clean ;;
        help|--help|-h) show_help ;;
        *) log_error "Comando desconhecido: $cmd"; show_help; exit 1 ;;
    esac
}

main "$@"
