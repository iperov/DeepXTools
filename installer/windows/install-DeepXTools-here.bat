@echo off

SET CD=%~dp0
SET TEMP_DIR=%CD%_temp\
SET TEMP_REPO=%TEMP_DIR%_repo\
SET TEMP_REPO_INSTALLER_PY=%TEMP_REPO%build\windows\installer.py

SET TEMP_PYTHON_ZIP=%TEMP_DIR%python.zip
SET TEMP_PYTHON_DIR=%TEMP_DIR%python\
SET TEMP_PYTHON_EXE=%TEMP_PYTHON_DIR%python.exe

if not exist "%TEMP_DIR%" (
  mkdir "%TEMP_DIR%"
)

powershell -Command Invoke-WebRequest https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip -OutFile "%TEMP_PYTHON_ZIP%"
powershell -Command Expand-Archive '%TEMP_PYTHON_ZIP%' -DestinationPath '%TEMP_PYTHON_DIR%' -Force 

rem TODO dl from repo
powershell -Command Remove-Item '%TEMP_REPO%' -Recurse -Force
powershell -Command Copy-Item 'D:\DevelopPPP\projects\DeepXTools\_internal\github_project' -Destination '%TEMP_REPO%' -Recurse -Force

"%TEMP_PYTHON_EXE%" "%TEMP_REPO_INSTALLER_PY%" --release-dir "%CD%DeepXTools" --cache-dir "%TEMP_DIR%_cache"

rmdir "%TEMP_DIR%" /s /q 2>nul
