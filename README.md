# LNBio ESPEM 2026 - Segmentação de imagens em tempo real com YOLOv8

Este guia explica, passo a passo, como preparar o computador e executar o programa de demonstração utilizado no LNBio ESPEM 2026. O programa realiza segmentação de imagens em tempo real utilizando a webcam do computador e o modelo pré-treinado YOLOv8.

## Requisitos do sistema

Antes de começar, verifique se você possui:

- **Python 3.12** (será instalado neste guia)
- **Computador ou notebook com webcam**
- **Placa de vídeo NVIDIA (opcional)**  
  > O programa funciona sem placa de vídeo dedicada, porém o desempenho será mais lento.

---

## 1. Instalando o Python no Windows

### 1.1 Baixar o Python

1. Clique [aqui](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) para fazer o download do Python 3.12.10

### 1.2 Instalar o Python

1. Abra o arquivo que você baixou (`python-3.12.10-amd64.exe`)

2. **Muito importante**: marque a opção  
   **Add Python to PATH**

3. Clique em **Install Now**

4. Aguarde a instalação finalizar

## 2. Baixando este projeto

1. Clique [aqui](https://github.com/r1ckhdk/espem-lnbio-2026/archive/refs/tags/v1.0.zip) para fazer o download do projeto compactado.

2. Extraia o arquivo

## 3. Criando o ambiente virtual e instalando as dependências

> Esta etapa garante que todas as bibliotecas necessárias sejam instaladas corretamente, sem interferir em outros programas do computador.

### 3.1 Abrir o terminal

1. Abra a pasta onde o projeto foi extraído

2. Clique com o botão direito em um espaço vazio da pasta

3. Selecione a opção “Abrir no Terminal” ou “Abrir no PowerShell”

### 3.2 Criar o ambiente virtual

No terminal, digite o comando abaixo e pressione Enter:

```bash
python -m venv venv
```

Esse comando cria um ambiente isolado chamado `venv`.

### 3.3 Ativar o ambiente virtual

Ainda no terminal, execute:

```bash
.\venv\Scripts\activate
```

Se deu certo, você verá algo como:

```bash
(venv)
```

no início da linha do terminal.

### 3.4 Instalar as dependências do projeto

Com o ambiente virtual ativado, execute:

```bash
(venv) pip install -r requirements.txt
```

Aguarde até que todas as bibliotecas sejam instaladas. Isso pode levar alguns minutos.

## 4. Executando o programa

Após concluir a instalação, execute o programa com o comando:

```bash
(venv) python live_inference.py
```

O que esperar ao executar:

- A webcam será ativada
- O programa abrirá em tela cheia
- A taxa de quadros por segundo (FPS) será exibida no canto superior esquerdo
- O sistema exibirá diferentes visualizações baseadas em inteligência artificial

## 5. Encerrando o programa

Para encerrar a execução do programa, pressione a tecla `q` no teclado.

# Dúvidas ou problemas?

Se você encontrar algum problema ou tiver dúvidas, sinta-se à vontade para abrir uma issue neste repositório em [Issues](https://github.com/cnpem/espem-lnbio-2026/issues).
