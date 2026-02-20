# Prod Docker deploy

## Что внутри
- `Dockerfile` — оптимизированный production-образ (multi-stage, non-root, минимальный runtime).
- `build_and_push_prod.ps1` — собирает образ, сохраняет `tar`, копирует на сервер `prod` в `/home/user/GROM/bitcoin_direction`.

## Запуск
Из корня проекта:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1
```

С кастомным тегом:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1 -Tag "2026-02-20"
```

С кастомным хостом/папкой:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1 -HostAlias "prod" -RemoteDir "/home/user/GROM/bitcoin_direction"
```

## Что выполнить на сервере после копирования

```bash
cd /home/user/GROM/bitcoin_direction
docker load -i xgb-bitcoin-direction-latest.tar
# запуск:
docker run -d --name xgb-bitcoin-direction -p 9743:9743 xgb-bitcoin-direction:latest
```

Если тег другой, имя архива будет `xgb-bitcoin-direction-<tag>.tar`.
