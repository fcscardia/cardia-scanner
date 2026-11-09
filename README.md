# CARDIA Scanner (v0.1)

Código de referência da prova de conceito **CARDIA | Prospect**. É **código de pesquisa, não de produção**.

## O que é

Uma ferramenta para ler os "estados internos" (ativações) de um modelo de linguagem de pesos abertos e testar se esses sinais ajudam a identificar quando o modelo **concorda indevidamente** com uma afirmação do usuário, por exemplo sob pressão ou insistência.

O CARDIA é uma **hipótese de pesquisa**, não uma solução comprovada. Este repositório existe para que a hipótese possa ser testada.

## O que o código faz

- Extrai as ativações do último token de textos, em camadas escolhidas (`extract`).
- Ajusta uma direção por diferença de médias entre dois conjuntos de textos (`fit_reference_direction`).
- Pontua textos novos pela projeção nessa direção (`score`).
- Escolhe a camada com melhor AUROC em dados separados do ajuste (`scan_layers`, `best_layer`).
- Calcula, token a token, uma projeção, a entropia e a variação do vetor interno (`scan_prompt`).
- Contém funções experimentais de intervenção nas ativações (`steer`, `ablate`) e de salvar/carregar direções.

## O que já foi testado

**Hipótese (notebooks de pesquisa).** Dois modelos pequenos, com afirmações factuais curtas em português (20 originais + 30 novas):

- **Gemma 2 2B IT:** a direção interna separa afirmações verdadeiras de falsas em afirmações novas (AUROC ≈ 0,86 no estado 14). A margem interna previu quais afirmações o modelo acaba aceitando sob pressão (teste confirmatório: correlação −0,72; AUROC 0,92).
- **Qwen 2.5 Coder 7B Instruct:** a direção separa verdadeiro de falso (AUROC ≈ 0,95 no estado 20), mas **não** previu a concordância indevida em afirmações novas (correlação −0,20; AUROC 0,63; critério não confirmado).

Notebooks no Kaggle:
- Gemma: https://www.kaggle.com/code/fcscardia/cardia-teste-de-dire-es-e-oda-id-no-gemma-2-2b
- Qwen: https://www.kaggle.com/code/fcscardia/cardia-teste-de-dire-es-e-oda-id-no-qwen2-5

**Código (este repositório).** Uma execução de verificação no Kaggle mostrou que o pacote instala, carrega um modelo Gemma de 2B, extrai ativações e calcula entropia e variação por token. Isso mostra que o código roda. **Não mostra** nenhum benefício de segurança, e não houve testes automatizados.

## O que NÃO foi demonstrado

- **O sinal interno não superou a pergunta direta ao modelo** ("a afirmação está correta? Sim/Não"), que teve desempenho comparável ou melhor nos dois modelos.
- **Nenhuma intervenção funcionou.** Remover (`ablate`) a direção de "pressão" ou reforçar (`steer`) a direção de "verdade" não reduziu a concordância indevida além de direções aleatórias, sem perder fluência ou respostas corretas. As funções de intervenção estão aqui só para experimentação e **não** foram validadas como proteção.
- O resultado preditivo usou a diferença entre a afirmação falsa e a sua versão verdadeira, **que não existe em produção**. Para um texto isolado, o sinal só foi validado para distinguir verdadeiro de falso.
- Os testes usam afirmações curtas e dois modelos pequenos. A transferência para conversas reais de atendimento é uma pergunta em aberto.
- A medição de concordância depende de um classificador de texto simples, sem validação humana.
- **Não foram testados:** uso concorrente (vários usuários ao mesmo tempo), servidores de inferência em lote contínuo (como vLLM), impacto em latência, nem modelos acessados só por API (sem acesso às ativações).

## Instalação

```bash
pip install -r requirements.txt
```

Para modelos de pesos abertos, use GPU. Detalhes em [INTEGRATION.md](INTEGRATION.md).

## Exemplo de uso

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cardia_scanner import CardiaTensorScanner

nome = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(nome)
model = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.float16).to("cuda")

scanner = CardiaTensorScanner(model, tokenizer)

# O polo "positivo" é o que você quer detectar (aqui, afirmação falsa).
# Em uso real, use dezenas de pares variados.
falsas      = ["a Terra é plana", "2 + 2 = 5"]
verdadeiras = ["a Terra é aproximadamente esférica", "2 + 2 = 4"]
camadas = [10, 12, 14, 16, 20]

# 1) Ajuste a direção
scanner.fit_reference_direction(falsas, verdadeiras, camadas)

# 2) Escolha a camada em dados SEPARADOS dos usados no ajuste
textos_val  = ["Paris é a capital da Alemanha", "Paris é a capital da França"]
rotulos_val = [1, 0]   # 1 = polo positivo (falsa), 0 = verdadeira
camada = scanner.best_layer(textos_val, rotulos_val, camadas)

# 3) Pontue um texto novo (maior = mais parecido com o polo positivo)
print(scanner.score(["o Brasil fica na Europa"], camada))
```

## Cuidados de uso

- **Escolha a camada para cada modelo.** A melhor camada muda de modelo para modelo; use `best_layer` com dados separados do ajuste.
- **Use dezenas de pares, variados.** Poucos pares dão resultados enganosamente bons.
- **A pontuação reflete o que o modelo "acha" da afirmação**, não a pressão do usuário em si.
- `scan_prompt` aplica a direção em todos os tokens, mas ela só foi validada no último token do prompt. Trate esse resultado como exploratório. Sem direção ajustada, a projeção vem vazia (`None`).
- O código aplica o template de chat do modelo e muda o `padding_side` do tokenizador para `left`.
- Em GPUs sem suporte nativo a `bfloat16` (como a T4), prefira `float16`.
- `ablate` e `steer` exigem uma direção já ajustada na camada pedida.

## Licença e contato

Licença indicada no arquivo `LICENSE`. Autor: Francisco Cardia. Críticas, replicações e resultados negativos são bem-vindos.
