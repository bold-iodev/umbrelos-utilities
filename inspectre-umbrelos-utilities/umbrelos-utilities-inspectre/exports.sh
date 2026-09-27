# Sourced by Umbrel. Architecture-specific upstream indices are pinned below.
export APP_UMBRELOS_UTILITIES_INSPECTRE_DB_PASSWORD="$(derive_entropy "${app_entropy_identifier}-db-password")"
export APP_UMBRELOS_UTILITIES_INSPECTRE_JWT_SECRET="$(derive_entropy "${app_entropy_identifier}-jwt-secret")"
export APP_UMBRELOS_UTILITIES_INSPECTRE_PROBE_SECRET="$(derive_entropy "${app_entropy_identifier}-probe-secret")"
export APP_UMBRELOS_UTILITIES_INSPECTRE_DOLLAR='$'
case "$(uname -m)" in
  aarch64|arm64)
    export APP_UMBRELOS_UTILITIES_INSPECTRE_WEB_IMAGE="thefunkygibbon/inspectre-web:raspi@sha256:33fe27b9e42493e573b4e42bc5b4da7d79ae588e9be869dd36ff5a9f751ef2fb"
    export APP_UMBRELOS_UTILITIES_INSPECTRE_PROBE_IMAGE="thefunkygibbon/inspectre-probe:raspi@sha256:eda2b084a325bb6d270f8cdb82aab90c8db8df8fdb0f87010ae7e0ea4a47b028"
    export APP_UMBRELOS_UTILITIES_INSPECTRE_FRONTEND_IMAGE="thefunkygibbon/inspectre-frontend:raspi@sha256:7a491bdfe28fdbaac4d8ecb432400c2da1840957717eed83d4412ad1e606ee9c"
    ;;
  x86_64|amd64)
    export APP_UMBRELOS_UTILITIES_INSPECTRE_WEB_IMAGE="thefunkygibbon/inspectre-web:latest@sha256:336f352072f67f32b46be5634624cec5a2c03a369f8cc7bcadb1859a3fe062b1"
    export APP_UMBRELOS_UTILITIES_INSPECTRE_PROBE_IMAGE="thefunkygibbon/inspectre-probe:latest@sha256:1855cb0c1c7fbdefadc314cf8e416eec79a2493f70e9717f73ba49021fa31ea3"
    export APP_UMBRELOS_UTILITIES_INSPECTRE_FRONTEND_IMAGE="thefunkygibbon/inspectre-frontend:latest@sha256:bb2f5691af38233e134808b5dddfe2ab955679258970d329cd0cb95055663847"
    ;;
  *) echo "InSpectre requires ARM64 or AMD64" >&2; return 1 ;;
esac
