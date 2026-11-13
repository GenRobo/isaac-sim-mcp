@echo off
setlocal DisableDelayedExpansion

:: All routine Windows launches go through the bounded, attached supervisor.
:: Overrides name a Python executable only (not an executable plus arguments).
if defined ISAAC_GUARD_PYTHON goto :selected
if defined PYTHON (
    set "ISAAC_GUARD_PYTHON=%PYTHON%"
    goto :selected
)
if exist "%ProgramData%\miniconda3\python.exe" (
    set "ISAAC_GUARD_PYTHON=%ProgramData%\miniconda3\python.exe"
    goto :selected
)
where py.exe >nul 2>&1
if not errorlevel 1 goto :py_launcher
where python.exe >nul 2>&1
if not errorlevel 1 (
    set "ISAAC_GUARD_PYTHON=python.exe"
    goto :selected
)
echo ERROR: Python 3.10+ not found. Set ISAAC_GUARD_PYTHON to a Python executable.
exit /b 1

:selected
for %%I in ("%ISAAC_GUARD_PYTHON%") do if /i "%%~nI"=="pythonw" goto :gui_python
for %%I in ("%ISAAC_GUARD_PYTHON%") do if /i "%%~nI"=="pyw" goto :gui_python
if exist "%ISAAC_GUARD_PYTHON%" goto :run
where "%ISAAC_GUARD_PYTHON%" >nul 2>&1
if not errorlevel 1 goto :run
echo ERROR: Selected Python executable is unavailable.
exit /b 1

:run
"%ISAAC_GUARD_PYTHON%" "%~dp0isaac_launch_guard.py" -- %*
exit /b %errorlevel%

:py_launcher
py.exe -3 "%~dp0isaac_launch_guard.py" -- %*
exit /b %errorlevel%

:gui_python
echo ERROR: A console Python interpreter is required; pythonw/pyw is not allowed.
exit /b 1
