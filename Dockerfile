# Usa la imagen oficial de Python ligera
FROM python:3.10-slim

# Hugging Face Spaces exige ejecutar las aplicaciones con un usuario no-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Establece el directorio de trabajo
WORKDIR /app

# Copia los requerimientos e instala las dependencias
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el código fuente al contenedor
COPY --chown=user . /app

# Expone el puerto 7860 que exige Hugging Face
EXPOSE 7860

# Ejecuta Gunicorn con un timeout alto (120s) porque cargar TensorFlow puede ser lento
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-t", "120", "main:crear_app()"]
