param(
    [string]$HostAlias = "prod",
    [string]$RemoteDir = "/home/user/GROM/bitcoin_direction",
    [string]$ImageName = "xgb-bitcoin-direction",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

$DeployDir = Resolve-Path $PSScriptRoot
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ContextDir = Join-Path $DeployDir ".tmp_prod_context"
$ArchiveName = "$ImageName-$Tag.tar"
$ArchivePath = Join-Path $DeployDir $ArchiveName
$ImageRef = "${ImageName}:${Tag}"
$LatestRef = "${ImageName}:latest"

try {
    Write-Host "[1/6] Preparing build context..."
    if (Test-Path $ContextDir) {
        Remove-Item $ContextDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ContextDir | Out-Null

    $requiredFiles = @(
        "app.py",
        "predict.py",
        "simple_backtest.py",
        "requirements.txt"
    )

    foreach ($file in $requiredFiles) {
        $source = Join-Path $ProjectRoot $file
        if (-not (Test-Path $source)) {
            throw "Required file not found: $source"
        }
        Copy-Item $source -Destination $ContextDir
    }

    $modelsSource = Join-Path $ProjectRoot "models"
    if (-not (Test-Path $modelsSource)) {
        throw "Required directory not found: $modelsSource"
    }
    Copy-Item $modelsSource -Destination $ContextDir -Recurse

    Copy-Item (Join-Path $DeployDir "Dockerfile") -Destination (Join-Path $ContextDir "Dockerfile")

    Write-Host "[2/6] Building Docker image: $ImageRef"
    docker build --pull -t $ImageRef $ContextDir
    if ($Tag -ne "latest") {
        Write-Host "[2.1/6] Tagging image as latest: $LatestRef"
        docker tag $ImageRef $LatestRef
    }

    Write-Host "[3/6] Saving image archive: $ArchivePath"
    if (Test-Path $ArchivePath) {
        Remove-Item $ArchivePath -Force
    }
    if ($Tag -eq "latest") {
        docker save -o $ArchivePath $ImageRef
    } else {
        docker save -o $ArchivePath $ImageRef $LatestRef
    }

    Write-Host "[4/6] Creating remote directory: ${HostAlias}:$RemoteDir"
    ssh $HostAlias "mkdir -p $RemoteDir"

    Write-Host "[5/6] Uploading archive via SCP..."
    scp $ArchivePath "${HostAlias}:$RemoteDir/"

    Write-Host "[6/6] Done. On server run:"
    Write-Host "ssh $HostAlias"
    Write-Host "cd $RemoteDir"
    Write-Host "docker load -i $ArchiveName"
}
finally {
    Write-Host "Cleaning temporary context..."
    if (Test-Path $ContextDir) {
        Remove-Item $ContextDir -Recurse -Force
    }
}
