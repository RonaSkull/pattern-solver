# 🚀 GitHub Codespaces - ARC-AGI-3 V6.3

**Ambiente de desenvolvimento na nuvem com Docker pré-instalado.**

Zero instalação local. Funciona no browser.

---

## 🎯 Como Usar (3 Passos)

### 1. Publique no GitHub
```bash
git init
git add -A
git commit -m "ARC-AGI-3 V6.3 - Production Ready"
git branch -M main
git remote add origin https://github.com/seu-usuario/ARC_AGI_3.git
git push -u origin main
```

### 2. Abra no Codespaces
1. Acesse: `https://github.com/seu-usuario/ARC_AGI_3`
2. Clique no botão **"<> Code"**
3. Selecione **"Codespaces"** tab
4. Clique **"Create codespace on main"**

### 3. Execute a Validação
No terminal do Codespaces (já aberto automaticamente):

```bash
# Validar agente
./scripts/validate-docker.sh all 10

# Ou manualmente:
docker build -f Dockerfile.arc-agi3 -t arc-agi3-validator .
docker run --rm arc-agi3-validator --validate --episodes 10
```

---

## ✅ O que já vem configurado

| Recurso | Status |
|---------|--------|
| Python 3.12 | ✅ Pré-instalado |
| Docker | ✅ Pré-instalado |
| SDK ARC-AGI-3 | ✅ Instalado no post-create |
| Extensões VS Code | ✅ Python, Docker, Jupyter |
| Portas 8080/8081 | ✅ Forward automático |

---

## 🔧 Comandos Úteis no Codespaces

```bash
# Testar agente
python -c "from arc_genetic_baby_v6.agent_v6 import ARCGeneticBabyV6; print('✓ OK')"

# Build submission
docker build -f Dockerfile.arc-agi3 -t arc-agi3-builder --target builder .
docker run --rm -v $(pwd)/submission:/output arc-agi3-builder

# Validar
./scripts/validate-docker.sh all 10

# Submeter para Kaggle (configure API key primeiro)
export ARC_API_KEY="sua-key-aqui"
./scripts/validate-docker.sh submission v6.3-complete
```

---

## 📋 Configurar Kaggle no Codespaces

```bash
# 1. Criar diretório Kaggle
mkdir -p ~/.kaggle

# 2. Adicionar API key (cole o conteúdo do kaggle.json)
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username":"seu-username","key":"sua-key"}
EOF

# 3. Permissões
chmod 600 ~/.kaggle/kaggle.json

# 4. Testar
kaggle competitions list
```

---

## 💡 Vantagens do Codespaces

- ✅ **Grátis** para repositórios públicos (120h/mês)
- ✅ **Docker funciona** (docker-in-docker)
- ✅ **Python 3.12** pré-instalado
- ✅ **SDK ARC-AGI-3** já configurado
- ✅ **Zero setup** - abre e usa
- ✅ **Sincronizado** com GitHub automaticamente

---

## 🚨 Limitações Free

- 120 horas/mês de uso
- 2 cores, 8GB RAM, 32GB espaço
- Repositório deve ser **público**

**Para competição:** Mais que suficiente para validar e submeter.

---

## 🎯 Workflow Rápido

```bash
# 1. Push para GitHub
git push origin main

# 2. No Codespaces (browser):
./scripts/validate-docker.sh all 10

# 3. Download submission.zip
# Codespaces → Explorer → Download

# 4. Submeter no Kaggle (upload manual ou CLI)
```

---

**Pronto para usar!** 🏆
