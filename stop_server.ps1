# Stops the NeMo 30B Speech-LLM local server and frees VRAM
Write-Host "[*] Stopping NeMo Speech Server..." -ForegroundColor Yellow
 = Get-Process nemo-speech -ErrorAction SilentlyContinue
if () {
     | Stop-Process -Force
    Write-Host "[SUCCESS] Stopped nemo-speech. VRAM freed on RTX 4080." -ForegroundColor Green
} else {
    Write-Host "[*] No nemo-speech server process was running." -ForegroundColor Cyan
}
