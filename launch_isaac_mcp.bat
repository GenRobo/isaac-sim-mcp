@echo off
setlocal

:: Isaac Sim MCP Launcher
:: Starts Isaac Sim with the MCP extension enabled

:: --- Configuration ---
:: Set this to your Isaac Sim installation directory
set ISAAC_SIM_ROOT=D:\github\IsaacSim\_build\windows-x86_64\release

:: Path to this MCP extension folder
set MCP_EXT_FOLDER=%~dp0
:: ---------------------

echo ========================================
echo  Isaac Sim MCP Launcher
echo ========================================
echo.
echo Isaac Sim: %ISAAC_SIM_ROOT%
echo MCP Extension: %MCP_EXT_FOLDER%
echo.

:: Check if Isaac Sim exists
if not exist "%ISAAC_SIM_ROOT%\isaac-sim.bat" (
    echo ERROR: Isaac Sim not found at: %ISAAC_SIM_ROOT%
    echo Please update ISAAC_SIM_ROOT in this script.
    pause
    exit /b 1
)

:: Launch Isaac Sim with MCP extension
echo Starting Isaac Sim with MCP extension...
cd /d "%ISAAC_SIM_ROOT%"
call isaac-sim.bat --ext-folder "%MCP_EXT_FOLDER%" --enable isaac.sim.mcp_extension

endlocal
