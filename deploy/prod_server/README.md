# Prod Docker Deploy

## Contents
- `Dockerfile` - optimized production image (multi-stage, non-root, minimal runtime).
- `build_and_push_prod.ps1` - builds the image, saves it as `tar`, and uploads it to `prod:/home/user/GROM/bitcoin_direction`.

## Run
From the project root:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1
```

With a custom tag:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1 -Tag "2026-02-20"
```

With a custom host/remote directory:

```powershell
.\deploy\prod_server\build_and_push_prod.ps1 -HostAlias "prod" -RemoteDir "/home/user/GROM/bitcoin_direction"
```

## What to run on the server after upload

```bash
cd /home/user/GROM/bitcoin_direction
docker load -i xgb-bitcoin-direction-<tag>.tar
# start container:
docker run -d --name xgb-bitcoin-direction -p 9743:9743 xgb-bitcoin-direction:latest
```

The build script now stores both tags inside the archive:
- `xgb-bitcoin-direction:<tag>`
- `xgb-bitcoin-direction:latest`

You can also run the exact tag directly:

```bash
docker run -d --name xgb-bitcoin-direction -p 9743:9743 xgb-bitcoin-direction:<tag>
```
