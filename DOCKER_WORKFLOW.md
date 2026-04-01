# 🚀 ARC-AGI-3 Genetic Baby V6.3 - Workflow Docker

## Resumo
Sistema de validação isolada em Docker para ARC-AGI-3. Zero conflito com seu Python 3.11.7.

---

## 📋 Arquivos Criados

| Arquivo | Descrição |
|---------|-----------|
| `Dockerfile.arc-agi3` | Imagem Docker com Python 3.12 + SDK oficial |
| `docker-compose.arc-validation.yml` | Orquestração de serviços de validação |
| `.env` | Configurações de ambiente (edite!) |
| `scripts/validate-docker.sh` | Script de automação |

---

## ⚡ Quick Start

### 1. Configurar
```bash
# Edite .env com seus dados
ARC_API_KEY=seu_api_key_aqui
ONLINE_MODE=false
```

### 2. Validar
```bash
# Tornar script executável (Git Bash ou WSL)
chmod +x scripts/validate-docker.sh

# Validação completa
./scripts/validate-docker.sh all 10
```

### 3. Submeter
```bash
# Se validação passar, submeter para Kaggle
kaggle competitions submit \
  -c arc-prize-2026-arc-agi-3 \
  -f submission/submission_v6.3-complete.zip \
  -m "v6.3-complete: Neuro-symbolic AGI w/ 11 gaps + 3 boom catalysts"
```

---

## 🐳 Comandos do Script

| Comando | Descrição |
|---------|-----------|
| `build` | Construir imagem Docker |
| `validate [n]` | Validar com n episódios |
| `test` | Rodar testes pytest |
| `submission [v]` | Criar submission.zip |
| `all` | Build + validate + submission |
| `clean` | Limpar containers |

---

## ✅ Pré-requisitos

- Docker Desktop instalado
- Docker Compose funcionando
- 4GB RAM disponível
- Conta Kaggle configurada (para submissão)

---

## 🔧 Solução de Problemas

### Docker não encontrado
```bash
# Verificar instalação
docker --version
docker compose version
```

### Permissão negada (Linux/Mac)
```bash
chmod +x scripts/validate-docker.sh
```

### Windows (PowerShell)
```powershell
# Usar diretamente o docker-compose
docker-compose -f docker-compose.arc-validation.yml build
docker-compose -f docker-compose.arc-validation.yml run arc-validator
```

---

## 📊 Arquitetura

```
Seu PC (Python 3.11.7)
    ↓ (desenvolve/testa)
Docker Container (Python 3.12 + SDK ARC-AGI-3)
    ↓ (valida em ambiente isolado)
Kaggle (avaliação oficial)
```

**Vantagem**: Seu Python 3.11.7 fica intacto. Validação real com SDK oficial.

---

## 🎯 Próximos Passos

1. ✅ Editar `.env` com `ARC_API_KEY`
2. ✅ Executar `./scripts/validate-docker.sh all 10`
3. ✅ Verificar `logs/validation_report.json`
4. ✅ Submeter para Kaggle se passar

---

**Pronto para produção.** 🏆
