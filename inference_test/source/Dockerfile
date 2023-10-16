# Use a imagem oficial do Python como base
FROM python:3.8

# Crie um diretório de trabalho dentro do container
WORKDIR /app

# upgrade pip
RUN pip install --upgrade pip

# Copie o arquivo de requisitos e instale as dependências
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copie os arquivos do projeto para o diretório de trabalho
COPY . .

# Exponha a porta em que a aplicação Flask irá rodar
EXPOSE 5000

# Inicie a aplicação com Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]