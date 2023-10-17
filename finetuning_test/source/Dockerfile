# Use a imagem base do Python
FROM python:3.10-slim

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie o script Python e o arquivo de requisitos
COPY code.py .
COPY requirements.txt .

# upgrade pip
RUN pip install --upgrade pip

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponha as pastas de entrada e saída
VOLUME /app/input
VOLUME /app/output

# Comando a ser executado quando o contêiner for iniciado
CMD ["python", "code.py"]


