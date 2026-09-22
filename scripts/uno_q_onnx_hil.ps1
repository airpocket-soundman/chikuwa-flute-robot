param(
    [string]$Adb = "C:\Users\yamas\platform-tools\adb.exe",
    [string]$ModelDir = "runs\yamabiko_composite_onnx",
    [string]$RemoteDir = "/home/arduino/yamabiko_onnx"
)
$ErrorActionPreference = "Stop"
$wheel = Get-ChildItem "runs\uno_q_onnx_wheels\onnxruntime*cp313*aarch64*.whl" | Select-Object -First 1
if (-not $wheel) { throw "Download an ONNX Runtime cp313 aarch64 wheel into runs/uno_q_onnx_wheels first." }
& $Adb shell "mkdir -p $RemoteDir/vendor $RemoteDir/model"
& $Adb push $wheel.FullName "$RemoteDir/onnxruntime.whl"
& $Adb shell "python3 -m zipfile -e $RemoteDir/onnxruntime.whl $RemoteDir/vendor"
foreach ($name in @("listen.onnx", "control_step.onnx", "demo_reference_frames.npy", "manifest.json")) {
    & $Adb push (Join-Path $ModelDir $name) "$RemoteDir/model/$name"
}
& $Adb push "scripts\bench_yamabiko_composite_onnx.py" "$RemoteDir/bench.py"
& $Adb shell "cd $RemoteDir && PYTHONPATH=vendor python3 bench.py model --out uno_q_report.json"
& $Adb pull "$RemoteDir/uno_q_report.json" "docs\e2e-composite-results\uno_q_onnx_report.json"
