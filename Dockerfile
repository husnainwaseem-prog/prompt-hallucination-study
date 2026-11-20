FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the model at build time (or mount it at runtime and pass --model).
# RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B-Instruct'); \
#     AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B-Instruct')"

CMD ["python", "-m", "src.cli", "run"]
