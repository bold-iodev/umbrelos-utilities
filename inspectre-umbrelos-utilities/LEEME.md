# InSpectre para UmbrelOS Utilities

Paquete comunitario experimental preparado para:
https://github.com/bold-iodev/umbrelos-utilities (rama `master`).

## Subir a tu tienda

1. Descomprime el ZIP.
2. Copia la carpeta **umbrelos-utilities-inspectre** completa a la raíz del repositorio, junto a `umbrel-app-store.yml` y `umbrelos-utilities-watermark-service`.
3. Incluye los archivos ocultos `.gitkeep` de `data/`. No subas el ZIP como sustituto de la carpeta.
4. Publica los archivos en la rama `master`. No necesitas modificar `umbrel-app-store.yml`.
5. Actualiza la tienda comunitaria en Umbrel, busca **InSpectre** e instálalo.

El icono se sirve desde tu repositorio y empezará a funcionar al publicar esa carpeta. No se ha subido nada a GitHub ni instalado nada en la Raspberry.

## Qué integra

- Icono y apertura desde Umbrel en el puerto 8799.
- Gestión de todos sus contenedores mediante los controles de iniciar/detener de Umbrel.
- Autenticación de Umbrel delante del panel y cuenta propia de InSpectre.
- PostgreSQL persistente bajo el directorio de datos de la app.
- Secretos distintos por instalación y finalidad, derivados por Umbrel.
- Selección de imágenes ARM64 o AMD64 con digest inmutable.
- Detección ARP, presencia y nombres. En una base de datos nueva se desactivan los escaneos automáticos de puertos, fingerprinting de servicios, reintentos y escaneos nocturnos. Los cambios posteriores desde la interfaz se conservan.

## Primer inicio

Crea la cuenta de InSpectre en su asistente. Revisa que detecta `end0` y `192.168.1.0/24` en tu Raspberry; las interfaces Docker y Tailscale no son la red doméstica. Si la detección automática elige otra red, corrígela en el asistente.

Mantén desactivados inicialmente los escaneos de vulnerabilidades/puertos. Asigna nombres a los dispositivos detectados y marca los conocidos. Las alertas en el móvil necesitan un canal configurado (por ejemplo, ntfy, Gotify o Telegram); el paquete no crea cuentas ni envía notificaciones a terceros.

La integración opcional con AdGuard se configura en InSpectre después de instalarlo. No se incluyen contraseñas de AdGuard, del router ni el token MCP.

## Puertos y permisos

- `8799`: panel web, publicado por el proxy de Umbrel.
- `127.0.0.1:15439`: PostgreSQL, accesible solo desde el host para el detector.
- `18666`: API del detector en la red del host. El código upstream escucha en todas las interfaces; sus operaciones requieren un secreto por instalación (excepto `/health`). No abrir este puerto en el router.
- Solo el detector usa `network_mode: host` y `CAP_NET_RAW`. No se usa modo privilegiado ni `NET_ADMIN`, y ningún servicio tiene el socket Docker de Umbrel.

La gestión/actualización de otros contenedores desde InSpectre y el bloqueo de dispositivos no están soportados en este paquete. No habilites funciones de interceptación de tráfico: este despliegue se ha diseñado para presencia y alertas. No ofrece estadísticas completas de GB de todos los dispositivos por estar conectado a la LAN.

## Validación y límites

Se han comprobado las referencias públicas de imágenes y sus plataformas, la sintaxis de Compose en ambas arquitecturas, los secretos, el cableado interno, las plantillas y el manifiesto. Consulta `VALIDACION.txt`.

**No se ha probado aún una instalación real en Umbrel**, ni la detección LAN, el inicio/parada o la persistencia en ejecución. El Docker local está apagado y no se dispone de acceso de instalación al Umbrel. Es un paquete para subir y probar, no una integración certificada.

Antes de darlo por estable:

1. Instalarlo desde la tienda y abrirlo desde el icono.
2. Completar el asistente y comprobar la detección de dispositivos.
3. Renombrar un dispositivo y configurar una alerta.
4. Detener e iniciar desde Umbrel; comprobar que conserva usuario, ajustes y nombres.
5. Verificar que AdGuard sigue funcionando y que no hay conflictos con los puertos 8799, 15439 y 18666.

No se han comprobado conflictos de puertos en tu Raspberry. Si alguno está ocupado, ajustar el paquete antes de instalar. La detección puede retrasarse en móviles dormidos o con MAC privada; las redes de invitados aisladas requieren acceso de red adicional.

## Mantenimiento

La versión `2026.09.27-1` identifica esta revisión del paquete y el conjunto de imágenes fijado, no una versión verificada dentro de cada imagen. Se revisó el código upstream en el commit `953d9c7b4c19d16c78e872feb39aba5e3d2d636e` (su archivo VERSION declara 1.2.53). No se ha acreditado que las imágenes publicadas correspondan exactamente a ese commit: verificar su versión al arrancar.

Upstream publica `raspi` (ARM64) y `latest` (AMD64), no un único índice multiarch; `exports.sh` selecciona índices fijados por digest. Se adapta así a una tienda comunitaria: no cumple literalmente el requisito de imagen única multiarch de la tienda oficial. El linter oficial no resuelve estas variables; las comprobaciones sin errores se realizaron sobre copias con las imágenes resueltas por arquitectura. No se han modificado las reglas del linter.

Para actualizar, revisar nuevos digests, confirmar la compatibilidad de `bootstrap.py.template` y `nginx.conf.template`, cambiar los seis índices en `exports.sh`, aumentar la versión del manifiesto y repetir las pruebas de reinicio y persistencia. No activar actualizaciones de contenedores fuera del ciclo de Umbrel.

El PostgreSQL está fijado a 15.19-alpine3.24. No cambiar de versión mayor sin migración.

Los avisos del linter sobre host networking y CAP_NET_RAW son necesarios para descubrir la LAN; `no-new-privileges` reduce privilegios. El icono se incluye porque es una tienda comunitaria. Los tags moving están fijados por digest porque upstream no ofrece etiquetas versionadas.

## Créditos

InSpectre: https://github.com/thefunkygibbon/InSpectre
Autor: thefunkygibbon. Icono y configuración nginx adaptados del proyecto.
Licencia upstream: AGPL-3.0; incluida en `LICENSE.upstream`.
