@echo off
setlocal

:: Isaac Sim MCP Launcher
:: Starts Isaac Sim with the MCP extension enabled

:: --- Configuration ---
:: Set this to your Isaac Sim installation directory
:: For Isaac Sim 5.x built from source:
set ISAAC_SIM_ROOT=D:\github\IsaacSim\_build\windows-x86_64\release

:: For Isaac Sim installed via Omniverse Launcher (uncomment and modify):
:: set ISAAC_SIM_ROOT=C:\Users\%USERNAME%\AppData\Local\ov\pkg\isaac-sim-5.1.0

:: Path to this MCP extension folder (auto-detected from script location)
set MCP_EXT_FOLDER=%~dp0
:: Remove trailing backslash
set MCP_EXT_FOLDER=%MCP_EXT_FOLDER:~0,-1%
:: ---------------------

echo ========================================
echo  Isaac Sim MCP Launcher
echo ========================================
echo.
echo Isaac Sim:     %ISAAC_SIM_ROOT%
echo MCP Extension: %MCP_EXT_FOLDER%
echo.

:: Check if Isaac Sim exists
if not exist "%ISAAC_SIM_ROOT%\isaac-sim.bat" (
    echo ERROR: Isaac Sim not found at: %ISAAC_SIM_ROOT%
    echo.
    echo Please update ISAAC_SIM_ROOT in this script to point to your Isaac Sim installation.
    echo.
    echo Common locations:
    echo   - Built from source: D:\github\IsaacSim\_build\windows-x86_64\release
    echo   - Omniverse Launcher: C:\Users\USERNAME\AppData\Local\ov\pkg\isaac-sim-X.X.X
    echo.
    pause
    exit /b 1
)

:: Check if MCP extension exists
if not exist "%MCP_EXT_FOLDER%\isaac.sim.mcp_extension" (
    echo ERROR: MCP extension not found at: %MCP_EXT_FOLDER%\isaac.sim.mcp_extension
    pause
    exit /b 1
)

:: Launch Isaac Sim with MCP extension
echo Starting Isaac Sim with MCP extension...
echo.
echo Command: isaac-sim.bat --ext-folder "%MCP_EXT_FOLDER%" --enable isaac.sim.mcp_extension
echo.
echo Once Isaac Sim loads, look for:
echo   "Isaac Sim MCP server started on localhost:8766"
echo.

cd /d "%ISAAC_SIM_ROOT%"
call isaac-sim.bat --ext-folder "%MCP_EXT_FOLDER%" --enable isaac.sim.mcp_extension %*

endlocal
