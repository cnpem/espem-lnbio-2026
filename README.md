# LNBio ESPEM 2026

## Requisitos

- Python 3.12
- Computador/notebook com webcam
- Placa de vídeo NVIDIA (opcional, mas desempenho será bem defasado sem uma placa de vídeo)

## 1. Instalando o Python no Windows

### 1.1 Baixar o Python

1. Clique [aqui](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) para fazer o download do Python 3.12.10

### 1.2 Instalar o Python

1. Abra o arquivo que você baixou (`python-3.12.10-amd64.exe`)
2. **Muito importante**: marque a opção  
   **Add Python to PATH**
3. Clique em **Install Now**
4. Aguarde a instalação finalizar

## 2. Fazendo o download do projeto

1. Clique [aqui](https://github.com/r1ckhdk/espem-lnbio-2026/archive/refs/tags/v1.0.zip) para fazer o download do projeto compactado.
2. Extraia o arquivo

## 3. Criando ambiente virtual e instalando dependências

1. Abra o terminal na pasta em que foi extraído o projeto, clicando com o botão direito em algum lugar vazio da pasta e selecionando a opção "Abrir no Terminal"
2. Crie o ambiente virtual com o comando: 
    ```bash
    python -m venv venv
    ```
3. Ative o ambiente virtual:
    ```bash
    .\venv\Scripts\activate
    ```
4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## 4. Executando o programa

Para executar o programa, rode o seguinte comando após ter instalado as dependências no passo anterior:

```bash
python live_inference.py
```
