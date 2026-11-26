@echo off
rem ===========================================================================
rem  Drill Failure Predict - offline environment installer (Windows x64)
rem
rem  Creates .venv (Python 3.10) in this folder from the shared wheel store
rem  E:\python_packages\, with NO internet access:
rem    1. Python 3.10      - uses an existing install, else installs
rem                          E:\python_packages\runtime\python-3.10.11-amd64.exe (current user)
rem    2. VC++ runtime     - installs E:\python_packages\runtime\vc_redist.x64.exe only if
rem                          msvcp140_1.dll is missing (needs admin)
rem    3. .venv            - python -m venv, then pip install --no-index from E:\python_packages\
rem                          exactly the versions in requirements-offline.txt
rem    4. checks           - tools\pip_check.py, tools\smoke_test.py, unit tests
rem
rem  The wheel store is shared by all projects. To use another location:
rem           set PYTHON_PACKAGES=D:\other\python_packages
rem
rem  Usage:   install_env.bat            install / repair
rem           install_env.bat /clean     delete .venv first, then install
rem           install_env.bat /test      only run the checks
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
rem shared offline wheel store (common to all projects); override with the PYTHON_PACKAGES variable
if not defined PYTHON_PACKAGES set "PYTHON_PACKAGES=E:\python_packages"
set "WHL=%PYTHON_PACKAGES%"
set "RT=%WHL%\runtime"
set "VENV=%ROOT%.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "REQ=%ROOT%requirements-offline.txt"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"

if /i "%~1"=="/test" goto :checks

set "WHL_OK=1"
if not exist "%WHL%\vtk-*.whl" set "WHL_OK="
if not exist "%WHL%\scipy-*.whl" set "WHL_OK="
if not exist "%WHL%\scikit_learn-*.whl" set "WHL_OK="
if not exist "%WHL%\lightgbm-*.whl" set "WHL_OK="
if not defined WHL_OK (
  echo [ERROR] shared wheel folder is missing or incomplete: %WHL%
  goto :fail
)
if not exist "%REQ%" (
  echo [ERROR] missing %REQ%
  goto :fail
)

rem ---------------------------------------------------------------- 1. Python 3.10
echo.
echo [1/4] Python 3.10
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY if exist "%ProgramFiles%\Python310\python.exe" set "PY=%ProgramFiles%\Python310\python.exe"
if defined PY goto :have_py
rem try the Windows "py" launcher (any other Python 3.10 install location)
where py >nul 2>&1 || goto :no_launcher
for /f "delims=" %%P in ('py -3.10 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%P"
:no_launcher
if defined PY goto :have_py
if not defined PY (
  if not exist "%RT%\python-3.10.11-amd64.exe" (
    echo [ERROR] Python 3.10 not found and %RT%\python-3.10.11-amd64.exe is missing.
    goto :fail
  )
  echo       not found - installing Python 3.10.11 for the current user ...
  "%RT%\python-3.10.11-amd64.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0
  if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
)
if not defined PY (
  echo [ERROR] Python 3.10 installation failed.
  goto :fail
)
:have_py
echo       using %PY%

rem ---------------------------------------------------------------- 2. VC++ runtime
echo.
echo [2/4] Microsoft Visual C++ 2015-2022 runtime
if exist "%SystemRoot%\System32\msvcp140_1.dll" (
  echo       already present
) else (
  echo       missing - running vc_redist.x64.exe ^(accept the UAC prompt^) ...
  "%RT%\vc_redist.x64.exe" /install /quiet /norestart
)

rem ---------------------------------------------------------------- 3. venv + wheels
echo.
echo [3/4] virtual environment %VENV% + packages ^(offline, from %WHL%^)
if /i "%~1"=="/clean" if exist "%VENV%\pyvenv.cfg" (
  echo       /clean: removing existing .venv ...
  rmdir /s /q "%VENV%"
)
if not exist "%VPY%" (
  "%PY%" -m venv "%VENV%"
  if errorlevel 1 goto :fail
)
"%VPY%" -m pip install --no-index --find-links "%WHL%" --upgrade pip --quiet
if %errorlevel% neq 0 goto :fail
"%VPY%" -m pip install --no-index --find-links "%WHL%" --no-deps -r "%REQ%"
if %errorlevel% neq 0 goto :fail

rem ---------------------------------------------------------------- 4. checks
:checks
echo.
echo [4/4] pip check + smoke test + unit tests
if not exist "%VPY%" (
  echo [ERROR] %VPY% not found - run install_env.bat without /test first.
  goto :fail
)
rem "neq 0", not "if errorlevel 1": a crash gives a negative exit code, which errorlevel 1 misses
"%VPY%" "%ROOT%tools\pip_check.py"
if %errorlevel% neq 0 goto :fail
"%VPY%" "%ROOT%tools\smoke_test.py"
if %errorlevel% neq 0 goto :fail
rem pytest's exit code is passed on with os._exit: Python's own shutdown after the GUI tests
rem (VTK / Qt teardown) can crash and would replace it with 0xC0000409
"%VPY%" -c "import os, sys, pytest; c = pytest.main(['-q', '-p', 'no:cacheprovider', sys.argv[1]]); sys.stdout.flush(); sys.stderr.flush(); os._exit(int(c))" "%ROOT%tests"
if %errorlevel% neq 0 goto :fail

echo.
echo DONE. Activate with:   %VENV%\Scripts\activate.bat     ^(cmd^)
echo                    or:  %VENV%\Scripts\Activate.ps1     ^(PowerShell^)
echo Start Drill Failure Predict:    run_dfpredict.bat
echo Build the release:      build_release.bat
endlocal
exit /b 0

:fail
echo.
echo *** INSTALL FAILED - see messages above ***
endlocal
exit /b 1
