# CARDIA Scanner (v0.1)

Framework de governança semântica e interpretabilidade para modelos de linguagem (LLMs), desenvolvido para a PoC CARDIA.

## Funcionalidades da v0.1
* Extração otimizada de ativações latentes (`CardiaTensorScanner`).
* Ajuste de direções de referência por Diferença de Centróides (`fit_reference_direction`).
* Análise de trajetória multi-token e entropia de Shannon.
* Intervenções seguras por gancho (*hooks*) com isolamento de contexto (`steer` e `ablate`).

## Instalação
```bash
pip install -r requirements.txt
