# Guia de integração (v0.1)

Este guia mostra como rodar o CARDIA Scanner com um modelo de pesos abertos do Hugging Face e quais são os limites atuais. É código de pesquisa, não de produção.

## 1. Requisitos

- Python 3.10 ou superior.
- Dependências em `requirements.txt` (torch, transformers, scikit-learn, numpy).
- **GPU recomendada.** Em CPU funciona, mas fica lento.
- Acesso aos **pesos** do modelo (modelo aberto). O scanner lê as ativações internas; **não funciona com modelos acessados apenas por API**.

Versões usadas na execução de verificação (Kaggle): transformers 5.20.2, scikit-learn 1.6.1, numpy 2.1.3. Outras versões não foram testadas.

## 2. Instalação

```bash
git clone https://github.com/fcscardia/cardia-scanner.git
cd cardia-scanner
pip install -r requirements.txt
```

## 3. Passo a passo

**3.1 Carregar o modelo.** Use `float16` em GPUs T4; evite `bfloat16` nelas. O scanner usa o mesmo tipo de dado do modelo.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cardia_scanner import CardiaTensorScanner

nome = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(nome)
model = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.float16).to("cuda")
scanner = CardiaTensorScanner(model, tokenizer)
```

Observação: o scanner altera `tokenizer.padding_side` para `left`.

**3.2 Montar os pares de contraste.** Duas listas de textos pareadas, uma por polo. O que é "positivo" depende da ordem que você usar em `fit_reference_direction(pos, neg, camadas)`. Use dezenas de pares, variados em tema e formato.

**3.3 Ajustar a direção e escolher a camada.**

```python
scanner.fit_reference_direction(pos_textos, neg_textos, camadas)
camada = scanner.best_layer(textos_val, rotulos_val, camadas)  # dados separados do ajuste
```

`rotulos_val`: 1 para o polo positivo, 0 para o negativo. O conjunto de validação precisa conter as duas classes.

**3.4 Pontuar textos novos.**

```python
scanner.score(["texto novo"], camada)   # maior = mais parecido com o polo positivo
```

**3.5 (Exploratório) Trajetória por token.**

```python
r = scanner.scan_prompt("texto", layer=camada)
# r["trajectory"]: token, risco_projetado, entropia_bits, velocidade
```

A direção só foi validada no último token do prompt; a projeção nos demais tokens é exploratória. O campo `risco_projetado` é apenas uma projeção, não uma probabilidade.

**3.6 (Experimental) Intervenções.** Exigem direção ajustada na camada pedida.

```python
with scanner.ablate(layer=camada):
    ...  # chamadas ao modelo aqui dentro têm a direção removida na camada
```

Não há evidência de que essas intervenções reduzam concordância indevida (ver README). Use só para pesquisa.

**3.7 Salvar e carregar direções.** `save_directions("arquivo.npz")` e `load_directions("arquivo.npz")`. A carga confere a dimensão do modelo, mas **não** verifica se é o mesmo modelo: use a direção só com o modelo em que foi ajustada.

**3.8 Encerrar.** `scanner.close()` remove os ganchos instalados no modelo.

## 4. Adaptar a outro modelo

O scanner procura as camadas em `model.layers`, `model.language_model.layers` ou `transformer.h`. Se a arquitetura for diferente, o código levanta erro: acrescente o caminho na lista `candidates` do método `_find_layers`. Só Gemma 2 e Qwen 2.5 foram usados nos experimentos de pesquisa.

## 5. Limites conhecidos

- Não testado com uso simultâneo (vários usuários) nem com `vLLM`/TGI. As intervenções atuam sobre todo o lote que passa pela camada, o que não é adequado a servidores com lote contínuo.
- Impacto em latência e memória não medido.
- A direção é uma diferença de médias simples; não foram comparados outros métodos neste repositório.
- Não há testes automatizados no repositório.
- Os resultados de pesquisa (README) valem para afirmações factuais curtas em dois modelos pequenos.

## 6. Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `Hierarquia de camadas não reconhecida` | Arquitetura com nome de camadas diferente | Ver seção 4 |
| `Geometria não treinada para a camada X` | Chamou `score` ou `ablate` sem ajustar a direção nessa camada | Rodar `fit_reference_direction` incluindo X |
| `labels precisa conter as duas classes` | Validação com uma só classe | Incluir exemplos 0 e 1 |
| Muito lento ou aviso sobre `bfloat16` | GPU sem suporte nativo (ex.: T4) ou CPU | Usar `float16` e GPU |
| Aviso sobre a última camada | O último estado é pós-normalização, o gancho age antes dela | Evitar a última camada em intervenções |
