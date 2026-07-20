@echo off
setlocal

set "ROOT=%~dp0..\.."
set "SOURCE=%ROOT%\sgl-kernel\csrc\evidence_sidecar\evidence_root_probe.cu"
set "OUTPUT=%~dp0evidence_root_probe.exe"
set "API_SOURCE=%ROOT%\sgl-kernel\csrc\evidence_sidecar\evidence_root_api.cu"
set "API_PROBE=%ROOT%\sgl-kernel\csrc\evidence_sidecar\evidence_root_api_probe.cu"
set "API_OUTPUT=%~dp0evidence_root_api_probe.exe"
set "API_LIBRARY=%~dp0sglang_evidence_root.dll"
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=sm_89"

set "VSDEVCMD=%ProgramFiles%\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
  echo Could not find VsDevCmd.bat. Install Visual Studio Build Tools with C++ support.
  exit /b 1
)

call "%VSDEVCMD%" -arch=x64
if errorlevel 1 exit /b %errorlevel%

nvcc --allow-unsupported-compiler -std=c++17 -O3 -lineinfo -Xptxas=-v -arch=%ARCH% "%SOURCE%" -o "%OUTPUT%"
if errorlevel 1 exit /b %errorlevel%
nvcc --allow-unsupported-compiler -std=c++17 -O3 -lineinfo -Xptxas=-v -arch=%ARCH% "%API_SOURCE%" "%API_PROBE%" -o "%API_OUTPUT%"
if errorlevel 1 exit /b %errorlevel%
nvcc --allow-unsupported-compiler -std=c++17 -O3 -lineinfo -Xptxas=-v -arch=%ARCH% -shared "%API_SOURCE%" -o "%API_LIBRARY%"
exit /b %errorlevel%
