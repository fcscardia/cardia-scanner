# Guia de Integração - CARDIA Scanner (v0.1)

Framework de governança semântica e interpretabilidade para modelos de linguagem (LLMs), desenvolvido para a PoC CARDIA.

## 1. Pré-requisitos de Ambiente

O ambiente de execução (seja local, Docker ou cluster Kubernetes) deve garantir as seguintes dependências base:

* **Python:** `>= 3.10`
* **PyTorch:** `torch >= 2.0.0` (suporte CUDA recomendado para testes de inferência rápida).
* **Hugging Face Transformers:** `transformers >= 4.38.0` (para compatibilidade com Gemma-2 e Llama-3).

## 2. Instalação na Infraestrutura

Para testes iniciais e pipelines CI/CD, o repositório pode ser clonado e as dependências instaladas via `pip`:

```bash
git clone https://github.com/fcscardia/cardia-scanner.git
cd cardia-scanner
pip install -r requirements.txt
```
## 3. Exemplo Prático de Integração Base

O módulo `cardia_scanner.py` foi desenhado para ser acoplado sem causar fricção na pipeline de inferência existente. Abaixo, o fluxo de inicialização:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cardia_scanner import CardiaTensorScanner

# 1. Carregar Modelo e Tokenizador
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 2. Inicializar o CARDIA Scanner
scanner = CardiaTensorScanner(model, tokenizer)

# 3. Executar Teste de Varredura (Scan Prompt)
prompt = "Defina Inteligência Artificial em uma frase curta."
print("Iniciando varredura das ativações latentes...")

activation_results = scanner.scan_prompt(prompt)

print(f"Camada monitorizada: {activation_results['camada']}")
print(f"Tokens analisados: {len(activation_results['trajectory'])}")
