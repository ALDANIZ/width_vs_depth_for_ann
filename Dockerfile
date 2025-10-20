FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# CPU versiyonu için pip install
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Veya requirements.txt'den kurulum
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
