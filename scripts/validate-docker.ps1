# ARC-AGI-3 Genetic Baby V6.3 - Windows Validation Script
# Validação completa em Docker para Windows (PowerShell)
# Uso: .\scripts\validate-docker.ps1 [comando] [opcoes]

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Arg1 = "",
    
    [Parameter(Position=2)]
    [string]$Arg2 = ""
)

# Cores para output
$Blue = "`e[34m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Reset = "`e[0m"

function Write-Info { param([string]$msg) Write-Host "${Blue}[INFO]${Reset} $msg" }
function Write-Success { param([string]$msg) Write-Host "${Green}[OK]${Reset} $msg" }
function Write-Warning { param([string]$msg) Write-Host "${Yellow}[WARN]${Reset} $msg" }
function Write-Error { param([string]$msg) Write-Host "${Red}[ERR]${Reset} $msg" }

# Detectar comando docker compose correto
$DockerComposeCmd = $null
function Get-DockerComposeCommand {
    if ($DockerComposeCmd) { return $DockerComposeCmd }
    
    # Testar 'docker compose' (novo)
    try {
        $null = docker compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $script:DockerComposeCmd = "docker compose"
            return $script:DockerComposeCmd
        }
    } catch {}
    
    # Testar 'docker-compose' (antigo)
    try {
        $null = docker-compose version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $script:DockerComposeCmd = "docker-compose"
            return $script:DockerComposeCmd
        }
    } catch {}
    
    return $null
}

function Test-Prerequisites {
    Write-Info "Verificando prerequisitos..."
    
    # Docker
    try {
        $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
        if (-not $dockerVersion) {
            Write-Error "Docker nao esta rodando. Inicie o Docker Desktop."
            return $false
        }
        Write-Success "Docker: $dockerVersion"
    } catch {
        Write-Error "Docker nao encontrado. Instale o Docker Desktop."
        return $false
    }
    
    # Docker Compose
    $composeCmd = Get-DockerComposeCommand
    if (-not $composeCmd) {
        Write-Error "Docker Compose nao encontrado"
        return $false
    }
    Write-Success "Docker Compose: $composeCmd"
    
    # Arquivos necessarios
    $requiredFiles = @("Dockerfile.arc-agi3", "docker-compose.arc-validation.yml", "requirements.txt")
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-Error "Arquivo nao encontrado: $file"
            return $false
        }
    }
    Write-Success "Todos os arquivos necessarios presentes"
    
    return $true
}

function Invoke-DockerCompose {
    param([string[]]$Arguments)
    
    $composeCmd = Get-DockerComposeCommand
    if (-not $composeCmd) {
        Write-Error "Docker Compose nao disponivel"
        return $false
    }
    
    $cmdArgs = "-f docker-compose.arc-validation.yml $Arguments"
    $fullCmd = "$composeCmd $cmdArgs"
    
    Write-Info "Executando: $fullCmd"
    Invoke-Expression $fullCmd
    
    return $LASTEXITCODE -eq 0
}

function Build-Image {
    Write-Info "Construindo imagem Docker..."
    
    if (Invoke-DockerCompose "build arc-validator") {
        Write-Success "Imagem construida com sucesso"
        return $true
    } else {
        Write-Error "Falha ao construir imagem"
        return $false
    }
}

function Run-Validation {
    param([int]$Episodes = 10)
    
    Write-Info "Validando com $Episodes episodios..."
    
    # Criar diretorios
    New-Item -ItemType Directory -Force -Path "logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null
    New-Item -ItemType Directory -Force -Path "submission" | Out-Null
    
    $env:VALIDATION_EPISODES = $Episodes
    
    if (Invoke-DockerCompose "run --rm arc-validator") {
        Write-Success "Validacao concluida"
        
        # Mostrar relatorio se existir
        if (Test-Path "logs\validation_report.json") {
            Write-Info "Relatorio de validacao:"
            Get-Content "logs\validation_report.json" -Raw | ConvertFrom-Json | Format-List
        }
        return $true
    } else {
        Write-Error "Validacao falhou"
        return $false
    }
}

function Build-Submission {
    param([string]$Version = "v6.3-complete")
    
    Write-Info "Construindo submission $Version..."
    
    New-Item -ItemType Directory -Force -Path "submission" | Out-Null
    
    $env:BUILD_VERSION = $Version
    
    if (Invoke-DockerCompose "run --rm arc-builder") {
        Write-Success "Submission criado: submission\submission_$Version.zip"
        
        # Verificar tamanho
        $zipPath = "submission\submission_$Version.zip"
        if (Test-Path $zipPath) {
            $size = (Get-Item $zipPath).Length / 1MB
            if ($size -lt 500) {
                Write-Success "Tamanho OK: $([math]::Round($size, 2)) MB < 500MB"
            } else {
                Write-Warning "Tamanho excede limite: $([math]::Round($size, 2)) MB >= 500MB"
            }
        }
        return $true
    } else {
        Write-Error "Falha ao construir submission"
        return $false
    }
}

function Run-Tests {
    Write-Info "Executando testes..."
    
    if (Invoke-DockerCompose "run --rm arc-tester") {
        Write-Success "Testes concluidos"
        return $true
    } else {
        Write-Error "Testes falharam"
        return $false
    }
}

function Start-Dashboard {
    Write-Info "Iniciando dashboard..."
    
    if (Invoke-DockerCompose "up -d arc-dashboard") {
        Write-Success "Dashboard iniciado em http://localhost:8081"
        Write-Info "Para parar: .\scripts\validate-docker.ps1 stop-dashboard"
    }
}

function Stop-Dashboard {
    Write-Info "Parando dashboard..."
    Invoke-DockerCompose "down arc-dashboard"
    Write-Success "Dashboard parado"
}

function Clean-Environment {
    Write-Warning "Limpando containers e imagens..."
    Invoke-DockerCompose "down -v --remove-orphans" | Out-Null
    docker rmi -f arc-agi3-validator:latest arc-agi3-builder:latest 2>$null | Out-Null
    Write-Success "Limpeza concluida"
}

function Show-Help {
    @"
🚀 ARC-AGI-3 V6.3 - Validacao Docker para Windows

Uso: .\scripts\validate-docker.ps1 <comando> [opcoes]

Comandos:
  build              Construir imagem Docker
  validate [n]       Validar com n episodios (padrao: 10)
  test               Executar testes pytest
  submission [v]     Criar submission.zip (versao v)
  all [n]            Build + validate + submission
  dashboard          Iniciar dashboard web
  stop-dashboard     Parar dashboard
  clean              Limpar containers e imagens
  help               Mostrar esta ajuda

Exemplos:
  .\scripts\validate-docker.ps1 build
  .\scripts\validate-docker.ps1 validate 20
  .\scripts\validate-docker.ps1 all 10
  .\scripts\validate-docker.ps1 submission v6.3-complete

Variaveis de ambiente (.env):
  ARC_API_KEY        API key da competicao
  ONLINE_MODE        false=local, true=submissao
  VALIDATION_EPISODES Numero de episodios
"@
}

# Main
switch ($Command.ToLower()) {
    "build" {
        Test-Prerequisites | Out-Null
        Build-Image
    }
    "validate" {
        $episodes = if ($Arg1) { [int]$Arg1 } else { 10 }
        Test-Prerequisites | Out-Null
        Build-Image
        Run-Validation -Episodes $episodes
    }
    "test" {
        Test-Prerequisites | Out-Null
        Build-Image
        Run-Tests
    }
    "submission" {
        $version = if ($Arg1) { $Arg1 } else { "v6.3-complete" }
        Test-Prerequisites | Out-Null
        Build-Image
        Build-Submission -Version $version
    }
    "all" {
        $episodes = if ($Arg1) { [int]$Arg1 } else { 10 }
        $version = if ($Arg2) { $Arg2 } else { "v6.3-complete" }
        
        if (-not (Test-Prerequisites)) { exit 1 }
        if (-not (Build-Image)) { exit 1 }
        if (-not (Run-Validation -Episodes $episodes)) { exit 1 }
        if (-not (Build-Submission -Version $version)) { exit 1 }
        
        Write-Success "Workflow completo finalizado com sucesso!"
        Write-Info "Submission pronto: submission\submission_$version.zip"
    }
    "dashboard" {
        Test-Prerequisites | Out-Null
        Start-Dashboard
    }
    "stop-dashboard" {
        Stop-Dashboard
    }
    "clean" {
        Clean-Environment
    }
    "help" { Show-Help }
    default {
        Write-Error "Comando desconhecido: $Command"
        Show-Help
    }
}
