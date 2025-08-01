from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "images"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Procesa los archivos seleccionados (imágenes o vídeos) aplicando la marca de agua.

    Se admite una única marca de agua en formato PNG y una o varias imágenes/vídeos. La posición
    de la marca de agua se transmite como coordenadas relativas (0 a 1) mediante los campos
    pos_x y pos_y, para permitir al usuario colocarla libremente sobre la previsualización.
    """
    watermark_file = request.files.get('watermark')
    input_files = request.files.getlist('images')
    try:
        size_percentage = float(request.form.get('size_percentage', 50))
    except Exception:
        size_percentage = 50.0
    try:
        pos_x = float(request.form.get('pos_x', 0))
        pos_y = float(request.form.get('pos_y', 0))
    except Exception:
        pos_x = 0.0
        pos_y = 0.0

    # Opción para conservar audio en vídeos (1 = sí, 0 = no)
    keep_audio_flag = request.form.get('keep_audio_hidden', '0')
    keep_audio = str(keep_audio_flag) == '1'

    # Guardar la marca de agua temporalmente y convertir a RGBA
    watermark_path = os.path.join(UPLOAD_FOLDER, "watermark.png")
    watermark_file.save(watermark_path)
    watermark_img = Image.open(watermark_path).convert("RGBA")

    processed_files = []
    for file_storage in input_files:
        filename = file_storage.filename
        # Crear ruta de salida con misma extensión
        name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        # Procesamiento de imágenes
        if ext_lower in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            img = Image.open(file_storage).convert("RGBA")
            width, height = img.size
            # Redimensionar marca de agua manteniendo proporción
            new_width = int((size_percentage / 100.0) * width)
            ratio = watermark_img.height / watermark_img.width
            new_height = int(new_width * ratio)
            watermark_resized = watermark_img.resize((new_width, new_height))
            # Calcular posición absoluta
            pos_left = int(pos_x * width)
            pos_top  = int(pos_y * height)
            # Asegurar que la marca de agua no se salga del borde
            pos_left = max(0, min(pos_left, width - new_width))
            pos_top  = max(0, min(pos_top,  height - new_height))
            # Pegar la marca de agua
            img.paste(watermark_resized, (pos_left, pos_top), watermark_resized)
            img.save(output_path, format="PNG")
            processed_files.append(filename)
        # Procesamiento de vídeos
        elif ext_lower in ['.mp4', '.mov', '.mkv', '.avi', '.webm']:
            # Guardar vídeo temporalmente
            temp_video_path = os.path.join(UPLOAD_FOLDER, filename)
            file_storage.save(temp_video_path)
            # Procesar con OpenCV
            try:
                apply_watermark_to_video(temp_video_path, watermark_img, size_percentage, pos_x, pos_y, output_path)
                # Si se desea conservar el audio, mezclarlo con el vídeo original
                if keep_audio:
                    try:
                        merge_audio(temp_video_path, output_path)
                    except Exception as e:
                        print(f"Error combining audio for {filename}: {e}")
                processed_files.append(filename)
            except Exception as e:
                print(f"Error processing video {filename}: {e}")
        else:
            # Formato desconocido: omitir
            continue

    return jsonify({"files": processed_files})

def apply_watermark_to_video(video_path: str, watermark_img: Image.Image, size_percentage: float, pos_x: float, pos_y: float, output_path: str) -> None:
    """Aplica una marca de agua a cada frame de un vídeo utilizando OpenCV.

    La marca de agua se redimensiona según el porcentaje indicado del ancho del vídeo y se coloca
    en las coordenadas relativas (pos_x, pos_y). Actualmente, el audio no se conserva.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Convertir la marca de agua a numpy array y RGBA
    wm_rgba = np.array(watermark_img.convert('RGBA'))
    # Calcular tamaño de la marca de agua según el ancho del vídeo
    new_w = int((size_percentage / 100.0) * width)
    ratio = watermark_img.height / watermark_img.width
    new_h = int(new_w * ratio)
    wm_rgba_resized = cv2.resize(wm_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Separar canales
    wm_rgb  = wm_rgba_resized[:, :, :3]
    wm_alpha= wm_rgba_resized[:, :, 3] / 255.0
    # Calcular posición absoluta
    pos_left = int(pos_x * width)
    pos_top  = int(pos_y * height)
    pos_left = max(0, min(pos_left, width  - new_w))
    pos_top  = max(0, min(pos_top,  height - new_h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Asegurarse de tener canal alpha
        if frame.shape[2] == 3:
            base = frame.copy().astype(float)
        else:
            # Si el frame tiene alfa, ignorar
            base = frame[:, :, :3].copy().astype(float)
        # Extraer región de interés
        roi = base[pos_top:pos_top + new_h, pos_left:pos_left + new_w]
        # Combinación ponderada
        for c in range(3):
            roi[:, :, c] = (1.0 - wm_alpha) * roi[:, :, c] + wm_alpha * wm_rgb[:, :, c]
        base[pos_top:pos_top + new_h, pos_left:pos_left + new_w] = roi
        # Convertir a uint8
        out_frame = np.clip(base, 0, 255).astype('uint8')
        out.write(out_frame)
    cap.release()
    out.release()

def merge_audio(original_video_path: str, watermarked_video_path: str) -> None:
    """Combina la pista de audio del vídeo original con el vídeo resultante.

    Utiliza ffmpeg mediante subprocess. El vídeo resultante sobrescribe al archivo de entrada.
    Si ffmpeg no está disponible o ocurre un error, se lanzará una excepción y se conservará el vídeo sin audio.
    """
    import subprocess
    import tempfile
    # Generar un archivo temporal para la mezcla
    dir_name = os.path.dirname(watermarked_video_path)
    tmp_output = os.path.join(dir_name, f"tmp_{os.path.basename(watermarked_video_path)}")
    # Comando ffmpeg: copiar video desde el vídeo procesado y audio desde el original
    cmd = [
        'ffmpeg', '-y',
        '-i', watermarked_video_path,
        '-i', original_video_path,
        '-c:v', 'copy', '-c:a', 'aac',
        '-map', '0:v:0', '-map', '1:a:0',
        '-shortest', tmp_output
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Reemplazar el archivo de salida original con la versión con audio
        os.replace(tmp_output, watermarked_video_path)
    except subprocess.CalledProcessError as exc:
        # Si falla, eliminar temporal y propagar
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise RuntimeError(f"ffmpeg error combining audio and video: {exc.stderr.decode('utf-8', errors='ignore')}")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9009)
