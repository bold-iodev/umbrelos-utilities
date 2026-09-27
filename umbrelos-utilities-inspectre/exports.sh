# Sourced by Umbrel. Per-install secrets; image references must stay literal in Compose for Umbrel pull.
export APP_UMBRELOS_UTILITIES_INSPECTRE_DB_PASSWORD="$(derive_entropy "${app_entropy_identifier}-db-password")"
export APP_UMBRELOS_UTILITIES_INSPECTRE_JWT_SECRET="$(derive_entropy "${app_entropy_identifier}-jwt-secret")"
export APP_UMBRELOS_UTILITIES_INSPECTRE_PROBE_SECRET="$(derive_entropy "${app_entropy_identifier}-probe-secret")"
export APP_UMBRELOS_UTILITIES_INSPECTRE_DOLLAR='$'
