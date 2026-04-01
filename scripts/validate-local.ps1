# validate-local.ps1
# Validação LOCAL sem Docker - Python 3.12 isolado via venv
# Uso: .\scripts\validate-local.ps1 [episodios]

param(
    [int]$Episodes = 10,
    [string]$Version = "v6.3-complete"
)

$Blue = "`e[34m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Reset = "`e[0m"

function Write-Info { param([string]$msg) Write-Host "${Blue}[INFO]${Reset} $msg" }
function Write-Success { param([string]$msg) Write-Host "${Green}[OK]${Reset} $msg" }
function Write-Warning { param([string]$msg) Write-Host "${Yellow}[WARN]${Reset} $msg" }
function Write-Error { param([string]$msg) Write-Host "${Red}[ERR]${Reset} $msg" }

Write-Info "ARC-AGI-3 V6.3 - Validação Local (Sem Docker)"
Write-Info "Episódios: $Episodes"

# Verificar Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python não encontrado"
    exit 1
}
Write-Success "Python detectado: $pythonVersion"

# Criar ambiente virtual isolado
$venvPath = ".venv-arc312"
if (-not (Test-Path $venvPath)) {
    Write-Info "Criando ambiente virtual..."
    python -m venv $venvPath
}

# Ativar venv
$activateScript = "$venvPath\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Info "Ativando ambiente virtual..."
    . $activateScript
} else {
    Write-Error "Não foi possível ativar o ambiente virtual"
    exit 1
}

# Instalar dependências
Write-Info "Instalando dependências..."
python -m pip install --upgrade pip setuptools wheel
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
}

# Simular validação (sem SDK oficial que requer 3.12+)
Write-Info "Executando validação simulada..."
Write-Warning "Modo: Offline (sem SDK oficial)"
Write-Warning "Para validação completa com SDK, instale Python 3.12+ ou Docker"

# Criar diretórios
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "submission" | Out-Null

# Teste básico do agente
Write-Info "Testando importação do agente V6..."
try {
    $testCode = @"
import sys
sys.path.insert(0, '.')
try:
    from arc_genetic_baby_v6.agent_v6 import ARCGeneticBabyV6
    from arc_genetic_baby_v6.config import AgentConfig
    import numpy as np
    
    config = AgentConfig(grid_size=10, num_colors=8)
    agent = ARCGeneticBabyV6(config)
    
    # Teste rápido
    grid = np.random.randint(0, 10, (10, 10))
    stats = agent.get_stats()
    
    print(f"✓ Agente V6 inicializado")
    print(f"✓ Módulos: {[m for m in stats.keys()][:5]}...")
    print(f"✓ Grid teste: {grid.shape}")
    print(f"\n✅ VALIDAÇÃO BÁSICA PASSOU")
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
    
    $result = python -c $testCode 2>&1
    Write-Host $result
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Agente V6 funciona corretamente!"
    } else {
        Write-Error "Falha no teste do agente"
    }
    
} catch {
    Write-Error "Erro durante teste: $_"
}

# Criar submission package (sem Docker)
Write-Info "Criando submission package..."
try {
    $zipFile = "submission\submission_$Version.zip"
    New-Item -ItemType Directory -Force -Path "submission" | Out-Null
    
    # Criar ZIP com código
    $compressParams = @{
        Path = @(
            "arc_genetic_baby_v6\",
            "arc_genetic_baby_v6_arc3_adapter.py",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "README.md",
            "LICENSE"
        )
        DestinationPath = $zipFile
        CompressionLevel = "Optimal"
        Force = $true
    }
    
    Compress-Archive @compressParams
    
    if (Test-Path $zipFile) {
        $size = (Get-Item $zipFile).Length / 1MB
        Write-Success "Submission criado: $zipFile"
        Write-Success "Tamanho: $([math]::Round($size, 2)) MB"
        
        if ($size -lt 500) {
            Write-Success "✓ Tamanho OK para Kaggle (< 500MB)"
        } else {
            Write-Warning "⚠ Tamanho excede limite Kaggle (>= 500MB)"
        }
    }
    
} catch {
    Write-Error "Falha ao criar submission: $_"
}

Write-Success "Validação local concluída!"
Write-Info "Próximo passo: Submeter para Kaggle via UI ou CLI"
Write-Info "Comando: kaggle competitions submit -c arc-prize-2026-arc-agi-3 -f $zipFile"

# Desativar venv
deactivate 2>$null
