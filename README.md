# petguard-vision-ia-<APELLIDOS>

## Objetivo
Automatizar el registro y estandarización de imágenes veterinarias (fotos, radiografías, carnets) usando modelos de HuggingFace para extraer: especie, descripción, texto detectado y colores dominantes; y producir un JSON clínico listo para integración con sistemas de la clínica.

## Estructura
- `src/` : código Python
- `assets/` : imágenes usadas (poner `carnet_animal.jpg` aquí)
- `clinical_examples/` : JSON de ejemplo
- `requirements.txt`

## Instrucciones de ejecución (Spyder / consola)
1. Colocar `assets/carnet_animal.jpg` en la carpeta `assets/`.
2. Crear y activar entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\\Scripts\\activate    # Windows
