@echo off
rem ============================================================
rem  TopoSandbox 起動ランチャ
rem  イベント当日はこのファイルを実行する。
rem
rem  引数はそのまま渡される。
rem      run.bat              通常起動
rem      run.bat --replay     Kinect 無しで保存画像を再生
rem      run.bat --near       Near Mode
rem
rem  ※このファイルは Shift-JIS(CP932) + CRLF で保存すること。
rem    UTF-8 で保存すると cmd が日本語コメント行を解釈できず起動に失敗する。
rem ============================================================
setlocal

rem conda 環境名は本番機で稼働中のものをそのまま使っている。
call C:\ProgramData\miniforge3\condabin\conda.bat activate ar_sandbox
if errorlevel 1 goto :no_env

rem cupy が参照する CUDA を 12.8 に固定する。
rem マシン既定の CUDA_PATH は v11.8 を指しており、そのままでは
rem cupy-cuda12x が nvrtc64_120_0.dll を見つけられず起動に失敗する。
rem Python 3.8 以降は DLL 探索に PATH を使わないため、PATH 追加では解決しない。
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"
python -m topo_sandbox %*
if errorlevel 1 pause
endlocal
exit /b 0

:no_env
echo [ERROR] conda 環境 ar_sandbox を有効化できませんでした。
echo         README.md のセットアップ手順を確認してください。
pause
endlocal
exit /b 1
