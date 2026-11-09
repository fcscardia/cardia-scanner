# Guia de Integração: CARDIA Scanner (v0.1)

Este documento detalha o processo de integração da Prova de Conceito (PoC) do **CARDIA Scanner** na infraestrutura técnica da Blip, permitindo a extração de ativações latentes (L2 Stress) e a monitorização de Entropia de Shannon em modelos de linguagem (LLMs).

## 1. Pré-requisitos de Ambiente

O ambiente de execução da Blip (seja local, Docker ou cluster Kubernetes) deve garantir as seguintes dependências base:

* **Python:** `>= 3.10`
* **PyTorch:** `torch >= 2.0.0` (suporte CUDA recomendado para testes de inferência rápida).
* **Hugging Face Transformers:** `transformers >= 4.38.0` (para compatibilidade com Gemma-2 e Llama-3).

## 2. Instalação na Infraestrutura Blip

Para testes iniciais e pipelines CI/CD, o repositório pode ser clonado e as dependências instaladas via `pip`:

```bash
git clone [https://github.com/fcscardia/cardia-scanner.git](https://github.com/fcscardia/cardia-scanner.git)
cd cardia-scanner
pip install -r requirements.txt

## 3. Exemplo Prático de Integração Base

O módulo `cardia_scanner.py` foi desenhado para ser acoplado sem causar fricção na pipeline de inferência existente. Abaixo, o fluxo de inicialização e extração:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cardia_scanner import CardiaTensorScanner

# 1. Carregar Modelo e Tokenizador (Exemplo com arquitetura leve para testes)
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 2. Inicializar o CARDIA Scanner
# O scanner mapeará automaticamente a topologia das camadas ocultas.
scanner = CardiaTensorScanner(model, tokenizer)

# 3. Executar Teste de Varredura (Scan Prompt)
prompt = "Defina Inteligência Artificial em uma frase curta."
print("Iniciando varredura das ativações latentes...")

# A função scan_prompt retorna as ativações cruas (L2 Stress) e a Entropia
# ao longo de todas as camadas, sem impactar a geração de texto.
activation_results = scanner.scan_prompt(prompt)

print(f"Camadas analisadas: {len(activation_results.keys())}")
# Opcional: Identificar a camada com maior divergência estrutural
best_layer = scanner.best_layer(activation_results)
print(f"Camada candidata para Governança (ODA-ID): {best_layer}")
