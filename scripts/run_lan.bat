@echo off
rem ============================================================
rem  TopoSandbox 起動ランチャ（2 台構成の本番用・Topo-sandbox PC）
rem  イベント当日はこのファイルをダブルクリックする。
rem
rem  run.bat --stream ws://<3D 表示 PC>:8000/ws/ingest と同じ。
rem  砂場の高さを LAN 越しに 3D 表示 PC の topo-sandbox-live
rem  （向こうの scripts\run_lan.bat）へ送る。どちらを先に起動してもよい。
rem  ほかの引数はそのまま run.bat へ渡す（例: run_lan.bat --replay）。
rem
rem  3D 表示 PC の IP アドレスを変えたら、下の LIVE_PC も直すこと。
rem
rem  ※このファイルは Shift-JIS(CP932) + CRLF で保存すること。
rem    UTF-8 で保存すると cmd が日本語コメント行を命令と取り違える。
rem ============================================================
setlocal
set "LIVE_PC=192.168.10.2"
call "%~dp0run.bat" --stream ws://%LIVE_PC%:8000/ws/ingest %*
endlocal
