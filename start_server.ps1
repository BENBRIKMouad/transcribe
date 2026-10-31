# Starts the NeMo 30B Speech-LLM local server in the background
$serverUrl = "http://127.0.0.1:8080/health"
try {
    $res = Invoke-RestMethod -Uri $serverUrl -TimeoutSec 2 -ErrorAction Stop
    if ($res.status -eq "ok") {
        Write-Host "[*] NeMo 30B Server is already running and ready on port 8080." -ForegroundColor Green
        exit 0
    }
} catch {
    # Not running, proceed with starting
}

Write-Host "[*] Starting NeMo 30B Speech-LLM server in the background..." -ForegroundColor Cyan
$nemoBin = "C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\bin\nemo-speech.exe"
$model = "C:\Dev\nemo-speech-0.3.0-windows-x86_64-cuda\models\NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf"

$argsList = @(
    "serve",
    "--asr-model", $model,
    "--device", "cuda",
    "--host", "127.0.0.1",
    "--port", "8080",
    "--no-ui",
    "--asr.salm.max_sequences", "1",
    "--asr.salm.n_ctx_per_sequence", "28192",
    "--asr.salm.max_context", "28192",
    "--asr.salm.mtp_draft", "0",
    "--asr.salm.n_batch", "1024",
    "--asr.salm.encoder_max_batch", "8"
)

Start-Process -FilePath $nemoBin -ArgumentList $argsList -WindowStyle Hidden
Write-Host "[*] Server process started. Loading 30B model into RTX 4080..." -ForegroundColor Yellow

for ($i = 0; $i -lt 35; $i++) {
    Start-Sleep -Seconds 2
    try {
        $res = Invoke-RestMethod -Uri $serverUrl -TimeoutSec 1 -ErrorAction Stop
        if ($res.status -eq "ok") {
            Write-Host "`n[SUCCESS] NeMo 30B Server is ready on http://127.0.0.1:8080!" -ForegroundColor Green
            exit 0
        }
    } catch {
        Write-Host -NoNewline "."
    }
}
Write-Host "`n[!] Server started but still initializing. Check Task Manager." -ForegroundColor Yellow
