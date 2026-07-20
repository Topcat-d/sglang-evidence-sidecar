@echo off
setlocal

set "ROOT=%~dp0..\.."
set "SOURCE=%ROOT%\sgl-kernel\csrc\evidence_sidecar\evidence_root_probe.cu"
set "OUTPUT=%~dp0evidence_root_probe.exe"
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
exit /b %errorlevel%
