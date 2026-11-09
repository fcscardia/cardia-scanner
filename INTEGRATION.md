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
