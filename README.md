# HAI 22.04 — Diagnóstico geométrico normal-only em resíduos de RNN-AE

Repositório de reprodução do experimento com o **HAI 22.04**: reconstrução de resíduos assinados de uma RNN-AE monolítica, diagnóstico geométrico *normal-only* para Mahalanobis global e por processos, comparação principal com fusão MAX e análise exploratória de detectores locais isolados.

> **Estado da publicação:** o código e os resultados de referência foram selecionados e preservados. Os links para os conjuntos de dados no Google Drive e a validação da execução em uma cópia limpa ainda serão acrescentados. Não considerar este pacote integralmente reproduzido em outra máquina até concluir essas verificações.

## Escopo do experimento

- **Dataset:** HAI 22.04, 86 variáveis de processo, organizadas em P1 (44), P2 (24), P3 (7) e P4 (11).
- **Modelo de reconstrução:** RNN-AE monolítica, janela de 60 amostras, 64 unidades ocultas, uma camada recorrente, dropout 0, Adam com taxa de aprendizado `1e-3`, batch 512, semente 42, *early stopping* com paciência 5, máximo de 100 épocas.
- **Resíduo assinado:** `err = saída_RNN - entrada` no último instante da janela. O Mahalanobis é aplicado aos resíduos **assinados** (`err_`), não aos erros quadráticos (`res_`).
- **Ajuste estatístico:** Ledoit–Wolf com resíduos normais de treinamento e cálculo em `float64` no diagnóstico final.
- **Calibração:** quantil P99 da **validação exclusivamente normal**, separadamente para Mahalanobis global e para o máximo dos quatro escores locais. Os limiares são congelados antes da avaliação do teste.
- **Análise exploratória:** quatro detectores isolados, cada qual com seu P99 na validação normal; resultados não devem ser tratados como comparação principal pré-especificada.

## Arquivos do repositório

| Caminho | Finalidade |
| --- | --- |
| `codigo/rotular_testes_hai.py` | Adiciona identificadores de evento `Tipo` aos arquivos HAI. |
| `dataset_trabalho/hai-22.04/logs/candidatas_treino_validacao/candidata_A_segmentos.csv` | Mapa fixado dos intervalos de treino, guarda e validação. |
| `codigo/materializar_divisao_oficial_hai.py` | Materializa a divisão temporal segundo o mapa fixado. |
| `codigo/normalizar_hai.py` | Ajusta MinMax sobre os 806.046 registros do treino oficial e aplica a transformação a treino, validação e teste, preservando 17 booleanas. |
| `codigo/rnn_hai_auditada.py` | Define e treina a RNN-AE monolítica; gera checkpoint e escores originais. |
| `codigo/extrair_residuos_monolitico.py` | Extrai resíduos de validação/teste e confere os escores de teste contra a saída original da RNN. |
| `codigo/recuperar_residuos_treino_normal_only.py` | Recupera resíduos assinados de treino e confere parâmetros históricos do Mahalanobis. |
| `codigo/diagnosticar_hai_residuos_normal_only.py` | Calcula fatores geométricos e limiares normal-only. |
| `codigo/avaliar_hai_mahalanobis_global_local.py` | Avaliação principal de global versus local MAX nos ataques. |
| `codigo/analisar_hai_mahalanobis_por_grupo.py` | Análise exploratória das contribuições à MAX e dos grupos isolados. |
| `codigo/requirements.txt` | Congelamento das dependências do ambiente Linux empregado. |
| `resultado/rnn_hai/secaoMONOLITICO/modelo_treinado_rnn_secaoMONOLITICO.pth` | Checkpoint da RNN usada na avaliação. |
| `resultado/rnn_hai/secaoMONOLITICO/mahalanobis_global/parametros_mahalanobis_global.npz` | Parâmetros históricos, em precisão original, usados na conferência da recuperação dos resíduos. |
| `resultado/diagnostico_mahalanobis_normal_only/` | CSVs e JSONs finais de diagnóstico, comparação e análise exploratória. |

## Dados externos — links pendentes

Os arquivos grandes **não estão versionados no GitHub**. Serão disponibilizados no Google Drive, acompanhados de nomes, diretórios relativos e, quando possível, somas de verificação:

- **HAI 22.04 original:** seis arquivos `train1.csv`–`train6.csv` e quatro arquivos `test1.csv`–`test4.csv`, destinados a `dataset_trabalho/hai-22.04/`. **[INSERIR LINK DO GOOGLE DRIVE]**
- **Resíduos assinados de treino:** `resultado/diagnostico_mahalanobis_normal_only/residuos_treino_assinados.npy` (~265 MB). **[INSERIR LINK]**
- **Resíduos de validação:** `resultado/rnn_hai/secaoMONOLITICO/residuos_monolitico/residuos_validacao_monolitico.csv` (~410 MB). **[INSERIR LINK]**
- **Resíduos de teste:** `resultado/rnn_hai/secaoMONOLITICO/residuos_monolitico/residuos_teste_monolitico.csv` (~737 MB). **[INSERIR LINK]**
- **Escores originais da RNN:** `resultado/rnn_hai/secaoMONOLITICO/scores_rnn_secaoMONOLITICO.csv` (~26 MB); utilizados para a conferência do extrator. **[INSERIR LINK]**

A disponibilização dos dados deve observar suas condições de uso e atribuição. **Não execute programas de preparação sobre dados preexistentes sem backup:** alguns scripts históricos gravam saídas com `to_csv()` sem bloqueio de sobrescrita.

## Sequência de reprodução — preservar a referência

**Não executar programas diretamente na pasta de referência do GitHub.** Ela contém o checkpoint, os parâmetros históricos e os CSVs/JSONs publicados, que devem permanecer intactos. Alguns programas de preparação e treinamento escrevem resultados sem proteção de sobrescrita; os programas de diagnóstico e avaliação, ao contrário, recusam diretórios de saída existentes.

Use Python compatível com `codigo/requirements.txt`. O arquivo congela o ambiente Linux utilizado (Python 3.10.12, incluindo pacotes NVIDIA/CUDA); a instalação em outras plataformas ainda não foi validada. O procedimento abaixo não substitui uma execução de aceitação em instalação limpa, ainda pendente.

### 0. Preparar uma cópia de execução independente

Depois de obter o repositório, mantenha uma pasta como **referência** e crie uma **cópia de trabalho**. No exemplo, a pasta do repositório chama-se `hai_referencia`:

```bash
# Executar na pasta que contém hai_referencia. O subshell evita encerrar
# o terminal do usuário caso alguma verificação falhe.
(
  set -e
  test ! -e hai_execucao && test ! -L hai_execucao || {
    echo 'ERRO: hai_execucao ja existe. Nenhuma copia iniciada.'
    exit 1
  }
  cp -a hai_referencia hai_execucao
  cd hai_execucao

  # Preservar resultados publicados sem apagá-los.
  mkdir referencias_publicadas
  mv resultado/diagnostico_mahalanobis_normal_only/diagnostico referencias_publicadas/
  mv resultado/diagnostico_mahalanobis_normal_only/avaliacao_ataques referencias_publicadas/
  mv resultado/diagnostico_mahalanobis_normal_only/analise_exploratoria_por_grupo referencias_publicadas/
)&& cd hai_execucao
```

**Atenção:** os comandos `mv` acima devem ser usados **somente nessa cópia recém-criada**, nunca em `hai_referencia` nem na pasta histórica do pesquisador. Eles liberam os caminhos que os programas precisam criar e conservam os resultados publicados em `referencias_publicadas/` para comparação posterior. Não repita esse preparo sobre uma pasta de execução já utilizada.

### A. Reprodução principal a partir dos resíduos publicados (recomendada)

Obtenha do Google Drive os três artefatos intermediários e coloque-os nos caminhos relativos indicados em **Dados externos**: `residuos_treino_assinados.npy`, `residuos_validacao_monolitico.csv` e `residuos_teste_monolitico.csv`.

Na cópia `hai_execucao`, já preparada e com os diretórios de saída livres:

```bash
python codigo/diagnosticar_hai_residuos_normal_only.py
python codigo/avaliar_hai_mahalanobis_global_local.py
python codigo/analisar_hai_mahalanobis_por_grupo.py
```

O primeiro programa gera novamente os CSVs do diagnóstico. O segundo verifica os **hashes SHA-256 congelados** desses CSVs e os limiares P99 antes de consultar o teste. O terceiro verifica a reprodução do resultado principal e acrescenta as análises exploratórias. Compare os novos resultados com os arquivos preservados em `referencias_publicadas/` e com as métricas deste README. Se os hashes ou as verificações divergirem, **interrompa a reprodução e investigue o ambiente**; não modifique limiares, hashes ou dados para forçar concordância.

### B. Recriar os resíduos usando o checkpoint publicado

Este fluxo também requer uma cópia de trabalho independente, mas **não requer retreinar a RNN**. Coloque os dez CSVs originais do HAI na pasta indicada em **Dados externos** e mantenha o checkpoint e a referência histórica `.npz` que acompanham o repositório.

```bash
python codigo/rotular_testes_hai.py
python codigo/materializar_divisao_oficial_hai.py
python codigo/normalizar_hai.py
```

Depois, obtenha do Google Drive o arquivo `scores_rnn_secaoMONOLITICO.csv` no caminho indicado; ele é exigido pela auditoria do extrator de resíduos. Execute:

```bash
python codigo/extrair_residuos_monolitico.py
python codigo/recuperar_residuos_treino_normal_only.py
```

O extrator recria os resíduos da validação e do teste a partir do checkpoint e compara o escore global de teste com o arquivo histórico de escores. O recuperador recria os resíduos assinados do treinamento e os confronta com os parâmetros `.npz` históricos. **Não baixe previamente os três resíduos grandes para os mesmos caminhos de saída desse fluxo**: a recuperação de treino recusa sobrescrever o `.npy` existente, enquanto o extrator grava CSVs. Após concluir, siga as três instruções de avaliação do fluxo **A**.

### C. Novo treinamento completo — reprodução metodológica, não necessariamente numérica

Para estudar a reprodução a partir do treinamento, use **uma terceira cópia independente**, preservando a de referência e a de reprodução com checkpoint. Após rotular, dividir e normalizar, execute `python codigo/rnn_hai_auditada.py`. Esse programa grava um novo checkpoint e seus escores. Ele **não deve ser executado na pasta de referência ou na cópia destinada à reprodução exata**, pois substitui os pesos previamente publicados.

A semente 42 e os hiperparâmetros estão documentados, mas o treinamento em GPU/software diferente pode produzir pesos distintos. Consequentemente, a verificação contra o arquivo `.npz` histórico e os hashes do diagnóstico congelado **podem não ser satisfeitos**, ainda que o método tenha sido implementado corretamente. Uma avaliação de checkpoint recém-treinado é um **novo experimento**, cujos resultados devem ser identificados separadamente; não se deve alterar silenciosamente os verificadores originais.

A seleção exploratória que levou ao mapa `candidata_A_segmentos.csv` não precisa ser repetida: o mapa definitivo está versionado e é aplicado diretamente por `materializar_divisao_oficial_hai.py`.

## Resultados de referência

**Diagnóstico normal-only:** todos os quatro fatores pessimistas foram inferiores a 1; portanto, não houve certificado conservador de vantagem geométrica uniforme para os grupos. A ausência desse certificado **não prova equivalência** e **não garante nenhum resultado de detecção**.

| Grupo | Fator pessimista | Fator mediano | Fator otimista |
| --- | ---: | ---: | ---: |
| P1 | 0,001496 | 1,289977 | 3,761926 |
| P2 | 0,256982 | 0,834728 | 1,067258 |
| P3 | 0,000585 | 0,833331 | 1,120386 |
| P4 | 0,000605 | 0,727775 | 1,194856 |

Os limiares congelados foram **199.4393418878611** (global) e **135.92393580193848** (local MAX), cada um com 1.977 alarmes em 197.646 registros normais de validação.

**Comparação principal no teste**, com 360.964 amostras e 58 eventos:

| Métrica | Global | Local MAX |
| --- | ---: | ---: |
| Verdadeiros positivos | 5.981 | 3.214 |
| Falsos positivos | 3.794 | 1.852 |
| Recall | 49,72% | 26,72% |
| F1 | 0,548590 | 0,375994 |
| FPR no teste | 1,0873% | 0,5308% |
| Eventos detectados | **55/58** | **47/58** |

O global detectou oito eventos não detectados pela fusão MAX; a fusão MAX não detectou eventos exclusivos em relação ao global. Embora ambos tivessem o mesmo número de alarmes na **validação normal**, o número de falsos positivos no **teste** foi diferente; os limiares não foram reajustados no teste.

**Análise exploratória:** os detectores **P2** e **P4** isolados, com seus próprios limiares P99 de validação, detectaram os ataques **A209** e **A212**, que o global e a fusão MAX não detectaram. O P2 produziu respectivamente 9 e 5 alarmes nas amostras desses ataques; o P4, 1 e 1. Isso não demonstra superioridade geral dos detectores locais e deve ser interpretado junto com as respectivas taxas de falsos positivos.

As tabelas completas estão nos CSVs versionados em `resultado/diagnostico_mahalanobis_normal_only/`.

## Integridade do diagnóstico publicado

Os arquivos originais do diagnóstico normal-only tiveram os seguintes SHA-256:

```text
5d9713a1d89065585e66dc02dfb4a42646be646779b615c55430a29951430901  diagnostico_por_grupo.csv
ac87726c478f6273eb4d5971828b41270cbfda0b61637d767801da5d04bdaf1a  diagnostico_resumo.csv
```

Esses hashes também estão incorporados aos verificadores dos programas de avaliação. Alterar o conteúdo desses CSVs exige uma nova rodada de auditoria, não uma atualização silenciosa dos números.

## Limitações e transparência

1. O diagnóstico geométrico é baseado exclusivamente em dados normais; não utiliza os ataques para ajustar fatores ou limiares.
2. O indicador pessimista maior que 1 é um **critério geométrico conservador dentro das hipóteses do modelo**, não uma garantia empírica de melhor recall ou FPR.
3. A análise de grupos isolados é **exploratória**, planejada após conhecer o resultado da comparação principal; não constitui confirmação prospectiva da hipótese.
4. A equivalência estrita a partir de treinamento novo não está garantida em hardware/software distintos; o checkpoint publicado define a execução de referência.
5. Este README é um ponto de entrada. O manual completo de reprodução e a localização definitiva dos dados ainda serão adicionados antes da publicação final.
