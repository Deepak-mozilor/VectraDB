param(
  [int]$Dim = 64,
  [int]$N = 50000,
  [int]$TopK = 10,
  [int]$HttpPort = 8080,
  [int]$GrpcPort = 50051,
  [int]$Concurrency = 200,
  [int]$Requests = 100000,
  [string]$DataDir = "vectradb_data_bench64",
  [switch]$SkipBuild,
  [switch]$SkipPreload,
  [switch]$SkipGrpc,
  [switch]$SkipHttp
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

function Assert-Tool($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    Write-Warning "Missing tool: $name. Please install it and re-run."
    return $false
  }
  return $true
}

function Wait-Port($host, $port, $timeoutSec = 30) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $timeoutSec) {
    try {
      $client = New-Object System.Net.Sockets.TcpClient
      $iar = $client.BeginConnect($host, $port, $null, $null)
      $success = $iar.AsyncWaitHandle.WaitOne(500)
      if ($success -and $client.Connected) {
        $client.EndConnect($iar)
        $client.Close()
        return $true
      }
      $client.Close()
    } catch {}
  }
  return $false
}

# 1) Build server (release)
if (-not $SkipBuild) {
  Write-Host "[1/6] Building server (release)..." -ForegroundColor Cyan
  cargo build -p vectradb-server --release | Write-Host
}

# 2) Start server
Write-Host "[2/6] Starting server..." -ForegroundColor Cyan
if (Test-Path $DataDir) { Remove-Item -Recurse -Force $DataDir }
$serverExe = Join-Path $repoRoot 'target\release\vectradb-server.exe'
if (-not (Test-Path $serverExe)) { throw "Server binary not found at $serverExe" }

$serverArgs = @('-p', $HttpPort, '--grpc-port', $GrpcPort, '-d', $Dim, '-D', $DataDir)
$serverProc = Start-Process -FilePath $serverExe -ArgumentList $serverArgs -PassThru -WindowStyle Minimized

if (-not (Wait-Port '127.0.0.1' $GrpcPort 30)) { throw "gRPC port $GrpcPort did not open in time" }
if (-not (Wait-Port '127.0.0.1' $HttpPort 30)) { throw "HTTP port $HttpPort did not open in time" }
Write-Host "Server PID: $($serverProc.Id)" -ForegroundColor Green

try {
  # 3) Preload dataset
  if (-not $SkipPreload) {
    Write-Host "[3/6] Preloading $N vectors (dim=$Dim) via gRPC..." -ForegroundColor Cyan
    if (-not (Assert-Tool 'python')) { throw "Python not found" }
    python .\bench\preload.py --host 127.0.0.1 --port $GrpcPort --n $N --dim $Dim
  }

  # Prepare payload.json
  $vec = @(1.0) + (0..($Dim-2) | ForEach-Object { 0.0 })
  $payloadObj = @{ vector = $vec; top_k = $TopK }
  $payloadPath = Join-Path $repoRoot 'payload.json'
  $payloadObj | ConvertTo-Json -Compress | Set-Content -NoNewline -Path $payloadPath -Encoding UTF8

  # 4) gRPC throughput via ghz
  if (-not $SkipGrpc) {
    Write-Host "[4/6] Running gRPC benchmark with ghz..." -ForegroundColor Cyan
    if (Assert-Tool 'ghz') {
      $protoPath = Join-Path $repoRoot 'proto\vectradb.proto'
      $json = Get-Content $payloadPath -Raw
      ghz --insecure --proto $protoPath --call vectradb.VectraDb/SearchSimilar `
        -d $json -c $Concurrency -n $Requests "127.0.0.1:$GrpcPort"
    } else {
      Write-Warning "Skipping gRPC benchmark (ghz not found)."
    }
  }

  # 5) HTTP throughput via hey
  if (-not $SkipHttp) {
    Write-Host "[5/6] Running HTTP benchmark with hey..." -ForegroundColor Cyan
    if (Assert-Tool 'hey') {
      hey -n $Requests -c $Concurrency -m POST -H "Content-Type: application/json" `
        -D $payloadPath "http://127.0.0.1:$HttpPort/search"
    } else {
      Write-Warning "Skipping HTTP benchmark (hey not found)."
    }
  }

  Write-Host "[6/6] Benchmarks complete." -ForegroundColor Green
} finally {
  if ($serverProc -and -not $serverProc.HasExited) {
    Write-Host "Stopping server (PID=$($serverProc.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $serverProc.Id -Force
    Start-Sleep -Milliseconds 300
  }
}
