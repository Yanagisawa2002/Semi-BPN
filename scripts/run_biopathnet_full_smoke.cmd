@echo off
setlocal

cd /d "%~dp0.."
set "PATH=C:\Users\cgliu\.conda\envs\repurpose-nbfnet-gpu2;C:\Users\cgliu\.conda\envs\repurpose-nbfnet-gpu2\Library\bin;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONPATH=%CD%"
set "TORCH_EXTENSIONS_DIR=%CD%\.torch_extensions_msvc_vs"
set "CC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
set "CXX=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

C:\Users\cgliu\.conda\envs\repurpose-nbfnet-gpu2\python.exe -m mechrep.training.run_original_biopathnet -s 1024 -c configs\biopathnet_full_smoke.yaml
