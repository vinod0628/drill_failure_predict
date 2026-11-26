@echo off
rem Visible GUI walkthrough of the drilling-risk functions (tests\test_gui_walkthrough.py).
rem The main window opens on a COPY of demo_data\sb3_workspace; the mouse moves slowly over every step.
rem
rem Usage:   run_gui_test.bat            normal pace (1.2 s per step)
rem          run_gui_test.bat 2          slower (2 s per step)
rem          run_gui_test.bat 0.6        faster
rem Do not use the mouse or keyboard while it runs.
setlocal
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo .venv not found - run install_env.bat first.
  exit /b 1
)
set "DFP_GUI_TEST=1"
if not "%~1"=="" (set "DFP_GUI_SPEED=%~1") else (set "DFP_GUI_SPEED=1.2")
set "PYTHONUTF8=1"
echo GUI walkthrough - %DFP_GUI_SPEED% s per step. Do not touch the mouse or keyboard while it runs.
rem pytest's exit code is passed on with os._exit: Qt / VTK teardown at interpreter exit can crash otherwise
"%VPY%" -c "import os, sys, pytest; c = pytest.main(['-s', '-q', '-p', 'no:cacheprovider', 'tests/test_gui_walkthrough.py']); sys.stdout.flush(); os._exit(int(c))"
set "RC=%errorlevel%"
if "%RC%"=="0" (echo GUI WALKTHROUGH: PASSED) else (echo GUI WALKTHROUGH: FAILED ^(exit code %RC%^))
endlocal & exit /b %RC%
