#!/usr/bin/env python3

"""
RNN Autoencoder para detecção de anomalias no dataset HAI.

O programa permite executar diferentes modelos ou grupos de variáveis
mantendo a mesma arquitetura e os mesmos hiperparâmetros experimentais.

Fluxo previsto:
- carregamento de dados HAI previamente normalizados;
- criação de janelas temporais sem atravessar arquivos;
- treinamento da RNN Autoencoder;
- validação e seleção do melhor checkpoint;
- calibração do limiar com dados normais de validação;
- avaliação no conjunto de teste;
- geração de métricas, matrizes de confusão e arquivos de scores contínuos.

A estrutura de múltiplos modelos é preservada para permitir posteriormente
experimentos monolíticos, por grupos e estratégias de fusão.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import psutil
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import gc


# ============================================================================
# >>>>>> CONFIGURAÇÃO DOS MODELOS <<<<<<
# ============================================================================

MODELOS_ATIVOS = ['MONOLITICO']

# ============================================================================
# >>>>>> DIRETÓRIOS DO PROJETO <<<<<<
# ============================================================================

DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_PROJETO = os.path.dirname(DIRETORIO_SCRIPT)

# ============================================================================
# >>>>>> HIPERPARÂMETROS E ARQUIVOS DE CONFIGURAÇÃO <<<<<<
# ============================================================================
WINDOW_SIZE = 60
BATCH_SIZE = 512

BASE_DADOS = os.path.join(
    DIRETORIO_PROJETO,
    "dataset_trabalho",
    "hai-22.04",
    "normalizados_oficiais"
)

DIRETORIO_TREINO = os.path.join(
    BASE_DADOS,
    "treino"
)

DIRETORIO_VALIDACAO = os.path.join(
    BASE_DADOS,
    "validacao"
)

DIRETORIO_TESTE = os.path.join(
    BASE_DADOS,
    "teste"
)

TRAIN_FILES = [
    os.path.join(
        DIRETORIO_TREINO,
        "train1_treino_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TREINO,
        "train2_treino_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TREINO,
        "train3_treino_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TREINO,
        "train4_treino_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TREINO,
        "train5_treino_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TREINO,
        "train6_treino_normalizado.csv"
    ),
]

VALIDATION_FILES = [
    os.path.join(
        DIRETORIO_VALIDACAO,
        "train1_validacao_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_VALIDACAO,
        "train2_validacao_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_VALIDACAO,
        "train3_validacao_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_VALIDACAO,
        "train4_validacao_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_VALIDACAO,
        "train5_validacao_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_VALIDACAO,
        "train6_validacao_normalizado.csv"
    ),
]

TEST_FILES = [
    os.path.join(
        DIRETORIO_TESTE,
        "test1_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TESTE,
        "test2_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TESTE,
        "test3_normalizado.csv"
    ),
    os.path.join(
        DIRETORIO_TESTE,
        "test4_normalizado.csv"
    ),
]

RESULTADO_BASE = os.path.join(
    DIRETORIO_PROJETO,
    "resultado",
    "rnn_hai"
)

os.makedirs(RESULTADO_BASE, exist_ok=True)

TRAIN_MODEL = True
TAXA_APRENDIZADO = 1e-3
SEED = 42



RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS  = 1
RNN_DROPOUT     = 0.0

USE_EARLY_STOP = True   # ou False se quiser desativar o ealy stop
ES_PATIENCE = 5        # paciencia
MAX_EPOCHS = 100        # Limite superior de épocas
SAVE_BEST_MODEL = True  # salvar o melhor modelo (true) ou o último modelo (false)

USE_AUTO_THRESHOLD = True  # Se True, usa limiar automático baseado no percentil
AUTO_THRESHOLD_PERCENTILE = 0.99  # Percentil para definir o limiar automático



# ============================================================================
# >>>>>> FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO <<<<<<
# ============================================================================
def load_hai_file(file_path, features_secao):
    """
    Carrega um único arquivo HAI já normalizado.

    Nenhuma normalização, correção ou exclusão silenciosa é realizada.
    Qualquer inconsistência interrompe a execução.
    """

    data = pd.read_csv(
        file_path,
        encoding='utf-8'
    )

    data.columns = data.columns.str.strip()

    # Estrutura obrigatória do HAI pré-processado
    colunas_obrigatorias = ['timestamp', 'Attack', 'Tipo']

    faltantes = [
        c for c in colunas_obrigatorias
        if c not in data.columns
    ]

    if faltantes:
        raise ValueError(
            f"{file_path}: colunas obrigatórias ausentes: {faltantes}"
        )

    # Todas as features solicitadas precisam existir
    features_faltantes = [
        c for c in features_secao
        if c not in data.columns
    ]

    if features_faltantes:
        raise ValueError(
            f"{file_path}: features ausentes: {features_faltantes}"
        )

    # Preserva identificação temporal e rótulos
    preserved_data = data[
        ['timestamp', 'Tipo', 'Attack']
    ].copy()

    # Seleciona apenas as features usadas pelo modelo
    X = data[features_secao].copy()
    y = data['Attack'].copy()

    # Conversão estrita
    X = X.apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(y, errors='raise')

    # Nenhum NaN é permitido
    if X.isna().any().any():
        raise ValueError(
            f"{file_path}: foram encontrados NaN nas features."
        )

    if y.isna().any():
        raise ValueError(
            f"{file_path}: foram encontrados NaN nos rótulos."
        )

    # Nenhum infinito é permitido
    if not np.isfinite(
        X.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            f"{file_path}: foram encontrados valores infinitos."
        )

    # Attack deve conter somente 0 e 1
    valores_attack = set(y.unique())

    if not valores_attack.issubset({0, 1}):
        raise ValueError(
            f"{file_path}: valores inválidos em Attack: {valores_attack}"
        )

    # Os três objetos devem manter exatamente o mesmo alinhamento
    if not (
        len(X) == len(y) == len(preserved_data)
    ):
        raise ValueError(
            f"{file_path}: desalinhamento entre features e rótulos."
        )

    print(
        f"[LOAD] {os.path.basename(file_path)}: "
        f"{len(X)} linhas, {X.shape[1]} features"
    )

    return X, y, preserved_data


def create_sliding_window(data, window_size, batch_size, step=1):
    if step != 1:
        raise ValueError(
            "A versão HAI auditada utiliza step=1."
        )

    if len(data) < window_size:
        raise ValueError(
            "O conjunto possui menos amostras que WINDOW_SIZE."
        )

    block_size = batch_size

    # Converte o arquivo uma única vez para float32.
    data_array = data.to_numpy(dtype=np.float32, copy=True)
    data_tensor = torch.from_numpy(data_array)

    # Cria as janelas como views da mesma memória.
    # Não copia cada janela individualmente.
    windows = data_tensor.unfold(
        dimension=0,
        size=window_size,
        step=1
    ).permute(0, 2, 1)

    blocks = [
        windows[start:start + block_size]
        for start in range(0, windows.shape[0], block_size)
    ]

    return blocks

# ============================================================================
# >>>>>> MODELO RNN AUTOENCODER <<<<<<
# ============================================================================
class AnomalyRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=1, dropout=0.0):
        super(AnomalyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.rnn_encoder = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity='tanh'
        )
        self.rnn_decoder = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity='tanh'
        )
        self.fc_out = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        _, hidden_e = self.rnn_encoder(x)
        seq_len    = x.size(1)
        batch_size = x.size(0)

        # Prepara entrada do decoder
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_size, device=x.device)
        decoder_output, _ = self.rnn_decoder(decoder_input, hidden_e)
        x_reconstructed = self.fc_out(decoder_output)
        return x_reconstructed

# ============================================================================
# >>>>>> MÉTRICAS E OUTRAS FUNÇÕES <<<<<<
# ============================================================================
def check_system_info():
    print("Verificando informações do sistema...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        gpu_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        print(f"Usando GPU: {gpu_name} ({gpu_memory:.2f} GB de memória disponível)")
    else:
        print("Nenhuma GPU disponível. Usando CPU.")

    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"Memória RAM disponível: {ram:.2f} GB\n")
    versao_pt = torch.__version__
    print("Versão do PyTorch:", versao_pt)


def generate_histograms(anomaly_scores, y_true, threshold, output_folder):
    print("Gerando histogramas das pontuações de anomalia...")
    os.makedirs(output_folder, exist_ok=True)

    # Histograma Geral
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=False, label='Pontuações de Anomalia')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Limiar')

    plt.text(threshold, plt.ylim()[1] * 0.9,
         f'Limiar: {threshold:.6f}\n({AUTO_THRESHOLD_PERCENTILE * 100:.1f}%)',
         color='red', fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Pontuação de Anomalia')
    plt.ylabel('Frequência')
    plt.title('Distribuição Geral das Pontuações de Anomalia')
    plt.legend()
    plt.savefig(os.path.join(output_folder, "histograma_geral.png"))
    plt.close()

    # Normais
    normal_scores = np.array(anomaly_scores)[y_true == 0]
    plt.figure(figsize=(10, 6))
    sns.histplot(normal_scores, bins=50, kde=False, label='Pontuações de Normais', color='blue')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Limiar')

    plt.text(threshold, plt.ylim()[1] * 0.9,
         f'Limiar: {threshold:.6f}\n({AUTO_THRESHOLD_PERCENTILE * 100:.1f}%)',
         color='red', fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Pontuação de Anomalia')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Pontuações de Normais')
    plt.legend()
    plt.savefig(os.path.join(output_folder, "histograma_normal.png"))
    plt.close()

    # Anomalias
    anomaly_scores_only = np.array(anomaly_scores)[y_true == 1]
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores_only, bins=50, kde=False, label='Pontuações de Anomalias', color='orange')
    plt.axvline(x=threshold, color='red', linestyle='--', label='Limiar')

    plt.text(threshold, plt.ylim()[1] * 0.9,
         f'Limiar: {threshold:.6f}\n({AUTO_THRESHOLD_PERCENTILE * 100:.1f}%)',
         color='red', fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Pontuação de Anomalia')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Pontuações de Anomalias')
    plt.legend()
    plt.savefig(os.path.join(output_folder, "histograma_anomalias.png"))
    plt.close()

    # Combinado (Normais e Anomalias)
    plt.figure(figsize=(10, 6))
    sns.histplot(normal_scores, bins=50, kde=False, label='Normais', color='blue', alpha=0.6)
    sns.histplot(anomaly_scores_only, bins=50, kde=False, label='Anomalias', color='orange', alpha=0.6)
    plt.axvline(x=threshold, color='red', linestyle='--', label='Limiar')
    plt.xlabel('Pontuação de Anomalia')
    plt.ylabel('Frequência')
    plt.title('Distribuição Combinada (Normais x Anomalias)')
    plt.legend()
    plt.savefig(os.path.join(output_folder, "histograma_combinado.png"))
    plt.close()

    print("Histogramas salvos com sucesso.")


def create_consolidated_dataset(X_test, y_test, y_pred, preserved_data, output_folder):
    print("Criando dataset consolidado com previsões e rótulos reais...")
    tamanhos = {
        "X_test": len(X_test),
        "y_test": len(y_test),
        "y_pred": len(y_pred),
        "preserved_data": len(preserved_data),
    }

    if len(set(tamanhos.values())) != 1:
        raise ValueError(
            f"Desalinhamento em create_consolidated_dataset: {tamanhos}"
        )

    print(f"Tamanho validado: {len(X_test)}")

    consolidated_df = pd.DataFrame({
        "timestamp": preserved_data["timestamp"],
        "arquivo_origem": preserved_data["arquivo_origem"],
        "Tipo": preserved_data["Tipo"],
        "Rótulo_Real": y_test,
        "Rótulo_Previsto": y_pred
    })

    for col in X_test.columns:
        consolidated_df[col] = X_test[col].values


    # Mantém a ordem dos arquivos de teste e a ordem temporal interna.
    consolidated_df.sort_values(
        by=["arquivo_origem", "timestamp"],
        inplace=True
    )

    output_path = os.path.join(output_folder, "dataset_resultado_completo.csv")
    consolidated_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Dataset consolidado salvo em: {output_path}")

    return consolidated_df


def map_real_vs_predicted_attacks(output_folder):
    print("Gerando o arquivo de mapeamento de ataques reais e previstos...")
    consolidated_dataset_path = os.path.join(output_folder, "dataset_resultado_completo.csv")
    if not os.path.exists(consolidated_dataset_path):
        raise FileNotFoundError(f"Arquivo consolidado não encontrado em {consolidated_dataset_path}")

    test_df = pd.read_csv(consolidated_dataset_path)
    required_columns = [
        'arquivo_origem',
        'Tipo',
        'Rótulo_Real',
        'Rótulo_Previsto',
        'timestamp'
    ]
    for col in required_columns:
        if col not in test_df.columns:
            raise ValueError(f"A coluna '{col}' não está presente no DataFrame consolidado.")

    mapping_file_path = os.path.join(output_folder, 'mapa_atq_real_prev.txt')
    with open(mapping_file_path, 'w') as file:
        tipos_ataque = test_df['Tipo'].unique()
        for tipo in tipos_ataque:
            if tipo == 'Normal':
                continue
            ataques_reais = test_df[test_df['Tipo'] == tipo]
            if ataques_reais.empty:
                continue

            timestamp_inicio_real = ataques_reais['timestamp'].iloc[0]
            timestamp_fim_real    = ataques_reais['timestamp'].iloc[-1]
            num_linhas_real       = len(ataques_reais)

            file.write(f"Tipo de Ataque: {tipo}\n")
            file.write(f"Real: Início: {timestamp_inicio_real}, Fim: {timestamp_fim_real}, Linhas: {num_linhas_real}\n")

            ataques_previstos = ataques_reais[ataques_reais['Rótulo_Previsto'] == 1]
            if not ataques_previstos.empty:
                timestamp_inicio_prev = ataques_previstos['timestamp'].iloc[0]
                timestamp_fim_prev    = ataques_previstos['timestamp'].iloc[-1]
                num_linhas_prev       = len(ataques_previstos)
                file.write(f"Previsto: Início: {timestamp_inicio_prev}, Fim: {timestamp_fim_prev}, Linhas: {num_linhas_prev}\n")
            else:
                file.write(f"Previsto: Nenhum ataque detectado\n")

            file.write("-" * 50 + "\n")

    print(f"Arquivo de mapeamento de ataques salvo em: {mapping_file_path}")


def create_and_save_confusion_matrix(y_true, y_pred, output_folder):
    print("Criando a matriz de confusão...")
    conf_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
)
    total = conf_matrix.sum()

    # Cria os textos com valor absoluto e percentual
    annot = []
    for i in range(conf_matrix.shape[0]):
        row = []
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            percent = 100 * count / total if total > 0 else 0
            row.append(f"{count}\n({percent:.1f}%)")
        annot.append(row)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=annot, fmt="", cmap="Blues", cbar=False,
                     annot_kws={"size": 14, "color": "black"})
    plt.title("Matriz de Confusão")
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')

    # Colorir os quadrantes corretos e incorretos
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            color = 'lightgreen' if i == j else 'lightcoral'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color, lw=0))

    conf_matrix_path = os.path.join(output_folder, "matriz_confusao.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Matriz de confusão salva em: {conf_matrix_path}")
    return conf_matrix


def calculate_and_save_metrics(y_true, y_pred, anomaly_scores, output_folder):
    print("Calculando métricas...")
    f1 = f1_score(
        y_true, y_pred, average='binary', zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average='binary', zero_division=0
    )
    precision = precision_score(
        y_true, y_pred, average='binary', zero_division=0
    )

    if anomaly_scores is not None:
        roc = roc_auc_score(
            y_true,
            anomaly_scores
        )
    else:
        roc = None

    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metricas_teste_path = os.path.join(output_folder, 'metricas_teste.txt')
    with open(metricas_teste_path, 'w') as resumo:
        resumo.write(f"F1-Score: {f1:.4f}\n")
        resumo.write(f"Recall: {recall:.4f}\n")
        resumo.write(f"Precisão: {precision:.4f}\n")
        if roc is not None:
            resumo.write(f"ROC AUC: {roc:.4f}\n")
        else:
            resumo.write("ROC AUC: não calculado (score contínuo indisponível)\n")
        resumo.write(f"MCC: {mcc:.4f}\n")
        resumo.write(f"FPR: {fpr:.6f}\n")
        resumo.write(f"TN: {tn}\n")
        resumo.write(f"FP: {fp}\n")
        resumo.write(f"FN: {fn}\n")
        resumo.write(f"TP: {tp}\n")

    print(f"Métricas salvas em: {metricas_teste_path}")
    return {
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
        "ROC_AUC": roc,
        "MCC": mcc,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def save_descriptive_statistics(dataframe, output_folder):
    print("Salvando estatísticas descritivas...")
    test_stats = dataframe.describe().transpose()
    test_stats_path = os.path.join(output_folder, "test_stats.csv")
    test_stats.to_csv(test_stats_path)
    print(f"Estatísticas salvas em: {test_stats_path}")


def generate_test_results(
    y_true,
    y_pred,
    test_df,
    anomaly_scores,
    threshold,
    output_folder
):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    print(f"Tamanho de y_true: {len(y_true)}")
    print(f"Tamanho de y_pred: {len(y_pred)}")

    os.makedirs(output_folder, exist_ok=True)

    create_and_save_confusion_matrix(
        y_true,
        y_pred,
        output_folder
    )

    calculate_and_save_metrics(
        y_true,
        y_pred,
        anomaly_scores,
        output_folder
    )
    save_descriptive_statistics(test_df, output_folder)

    print("Gerando histogramas das pontuações de anomalia...")
    generate_histograms(
        anomaly_scores=anomaly_scores,
        y_true=y_true,
        threshold=threshold,
        output_folder=output_folder
    )


    map_real_vs_predicted_attacks(
        output_folder=output_folder
    )

def gera_relatorio_treino_secao(
    secao_id,
    epocas_efetivas,
    melhor_perda_validacao,
    perda_final_treino,
    perda_final_validacao,
    melhor_epoca,
    early_stop,
    salvamento_best,
    output_path
):
    """
    Gera um arquivo .txt com informações do treino para a seção especificada.

    Parâmetros:
    - secao_id (str): identificação do modelo ou grupo
    - epocas_efetivas (int): quantidade de épocas efetivamente treinadas
    - melhor_perda_validacao (float): menor perda observada na validação
    - perda_final_treino (float): perda de treino na última época executada
    - perda_final_validacao (float): perda de validação na última época executada
    - melhor_epoca (int): época em que ocorreu a menor perda de validação
    - early_stop (bool): indica se o early stopping estava ativo
    - salvamento_best (bool): indica se o melhor checkpoint foi preservado
    - output_path (str): caminho completo do relatório
    """

    # Cria o conteúdo do relatório
    relatorio_str = []
    relatorio_str.append("-------------------------------------------")
    relatorio_str.append(f"Relatório de Treino - Seção {secao_id}")
    relatorio_str.append("-------------------------------------------\n")


    relatorio_str.append(f"Épocas Efetivas (treinadas): {epocas_efetivas}")

    relatorio_str.append(
        f"Melhor perda de validação: {melhor_perda_validacao:.6f}"
    )
    relatorio_str.append(
        f"Perda final de treino: {perda_final_treino:.6f}"
    )
    relatorio_str.append(
        f"Perda final de validação: {perda_final_validacao:.6f}"
    )
    relatorio_str.append(
        f"Época de melhor validação: {melhor_epoca}"
    )

    relatorio_str.append(f"Early Stopping Ativado: {'SIM' if early_stop else 'NÃO'}")
    relatorio_str.append(f"Salvamento de Modelo: {'Melhor (SAVE_BEST_MODEL=TRUE)' if salvamento_best else 'Última Época (SAVE_BEST_MODEL=FALSE)'}\n")

    relatorio_str.append("Observações:")
    if early_stop and epocas_efetivas < MAX_EPOCHS:
        relatorio_str.append(
            "- O treinamento parou antecipadamente devido ao critério de Early Stopping."
        )
    elif early_stop:
        relatorio_str.append(
            "- O treinamento atingiu o número máximo de épocas sem acionar o Early Stopping."
        )
    else:
        relatorio_str.append(
            "- O treinamento rodou até o número máximo de épocas configurado."
        )
    if salvamento_best:

        relatorio_str.append(
            "- O arquivo do modelo salvo corresponde à menor perda de validação."
        )

    else:
        relatorio_str.append("- O arquivo do modelo salvo corresponde à última época de treinamento.")

    relatorio_str.append("-------------------------------------------\n")

    # Salva em arquivo
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in relatorio_str:
            f.write(line + "\n")

    print(f"Relatório de treino salvo em: {output_path}")

def gera_relatorio_geral(pasta_base, secoes_processadas):
    """
    Concatena os relatórios individuais das seções **processadas nesta execução**
    em um único relatório geral.

    Parâmetros:
        - pasta_base: Diretório onde os relatórios individuais estão armazenados.
        - secoes_processadas: Lista das seções processadas na rodada atual.
    """
    pasta_geral = os.path.join(pasta_base, "geral")
    os.makedirs(pasta_geral, exist_ok=True)  # Garante que a pasta existe

    relatorio_geral_path = os.path.join(pasta_geral, "relatorio_geral_treino.txt")

    with open(relatorio_geral_path, 'w', encoding="utf-8") as rel_geral:
        rel_geral.write("=== RELATÓRIO GERAL DO TREINAMENTO ===\n\n")
        rel_geral.write(f"Treinamento realizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        rel_geral.write(f"Número de seções processadas: {len(secoes_processadas)}\n\n")

        for secao_id in secoes_processadas:  # Apenas as seções desta rodada
            relatorio_secao_path = os.path.join(pasta_base, f"secao{secao_id}", f"relatorio_treino_secao{secao_id}.txt")

            if os.path.exists(relatorio_secao_path):
                with open(relatorio_secao_path, 'r', encoding="utf-8") as rel_secao:
                    rel_geral.write(f"=== SEÇÃO {secao_id} ===\n")
                    rel_geral.write(rel_secao.read())  # Copia o conteúdo do relatório da seção
                    rel_geral.write("\n" + "="*50 + "\n\n")  # Linha separadora
            else:
                rel_geral.write(f"=== SEÇÃO {secao_id} ===\n")
                rel_geral.write("ERRO: Relatório da seção não encontrado!\n\n")

    print(f"Relatório geral salvo em: {relatorio_geral_path}")



def gerar_relatorio_limiar(secao_id, limiar_usado, tipo_limiar, output_path):
    """
    Gera um relatório com o limiar usado na seção.
    """
    with open(output_path, 'w') as f:
        f.write(f"Relatório de Limiar - Seção {secao_id}\n")
        f.write("====================================\n")
        f.write(f"Limiar utilizado: {limiar_usado:.6f}\n")
        f.write(f"Tipo de limiar: {tipo_limiar}\n")

def atualizar_relatorio_global_limiar(geral_path, lista_de_relatorios):
    """
    Concatena todos os relatórios de limiar das seções processadas em um só.
    """
    with open(geral_path, 'w', encoding="utf-8") as f_geral:
        f_geral.write("Relatório Geral de Limiares\n")
        f_geral.write("====================================\n\n")

        for relatorio_path in lista_de_relatorios:
            if os.path.exists(relatorio_path):
                with open(relatorio_path, 'r', encoding="utf-8") as f_secao:
                    f_geral.write(f_secao.read() + "\n")
                    f_geral.write("-" * 50 + "\n")  # Linha separadora entre seções
            else:
                print(f"Aviso: Relatório não encontrado {relatorio_path}")

    print(f"Relatório global de limiares atualizado em: {geral_path}")

# ============================================================================
# >>>>>> FUNÇÃO PRINCIPAL COM LOOP DE SEÇÕES <<<<<<
# ============================================================================
def main():
    try:
        # Reprodutibilidade da inicialização experimental.
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        check_system_info()

        df_unificado = None
        relatorios_limiar = []


                # Descobre automaticamente as 86 features do HAI.
        colunas_hai = pd.read_csv(
            TRAIN_FILES[0],
            nrows=0,
            encoding='utf-8'
        ).columns.str.strip()

        features_hai = [
            c for c in colunas_hai
            if c not in ['timestamp', 'Attack', 'Tipo']
        ]

        if len(features_hai) != 86:
            raise ValueError(
                f"Esperadas 86 features HAI, encontradas {len(features_hai)}."
            )

        for secao_id in MODELOS_ATIVOS:

            if secao_id == 'MONOLITICO':
                features_secao = features_hai.copy()
            else:
                raise ValueError(
                    f"Modelo ainda não configurado: {secao_id}"
                )

            # Configurações da execução do modelo atual.
            limiar_secao = None
            num_epochs = MAX_EPOCHS

            print("\n============================================")
            print(
                f"*** MODELO {secao_id} *** -> "
                f"Máximo de épocas = {num_epochs}"
            )
            print("============================================\n")

            # Cria pasta específica para essa seção
            pasta_secao = os.path.join(RESULTADO_BASE, f"secao{secao_id}")
            os.makedirs(pasta_secao, exist_ok=True)

            # Nome do modelo
            model_file = os.path.join(pasta_secao, f"modelo_treinado_rnn_secao{secao_id}.pth")

            # Carrega os  arquivos de treino separadamente.
            # As fronteiras entre arquivos serão preservadas na criação das janelas.
            train_datasets = []

            for train_file in TRAIN_FILES:
                X_file, y_file, preserved_file = load_hai_file(
                    train_file,
                    features_secao
                )

                                # Os arquivos de treino HAI devem conter somente operação normal.
                if (y_file != 0).any():
                    raise ValueError(
                        f"{train_file}: arquivo de treino contém Attack != 0."
                    )

                train_datasets.append(
                    (train_file, X_file, y_file, preserved_file)
                )

            # Carrega os seis arquivos de validação separadamente.
            # A validação contém somente operação normal e será usada
            # para early stopping e calibração do limiar.
            validation_datasets = []


            for validation_file in VALIDATION_FILES:
                X_file, y_file, preserved_file = load_hai_file(
                    validation_file,
                    features_secao
                )

                if (y_file != 0).any():
                    raise ValueError(
                        f"{validation_file}: arquivo de validação contém Attack != 0."
                    )

                validation_datasets.append(
                    (validation_file, X_file, y_file, preserved_file)
                )


            print("Criando janelas deslizantes por arquivo...")

            # ============================================================
            # TREINO — cada arquivo gera suas próprias janelas.
            # Nunca existe janela atravessando train1 -> train2 etc.
            # ============================================================
                        # ============================================================
            # VALIDAÇÃO — cada arquivo gera suas próprias janelas.
            # Nunca existe janela atravessando os limites entre arquivos.
            # ============================================================
            X_validation_blocks = []

            for validation_file, X_file, y_file, preserved_file in validation_datasets:
                blocks = create_sliding_window(
                    X_file,
                    WINDOW_SIZE,
                    BATCH_SIZE
                )

                num_janelas = sum(block.shape[0] for block in blocks)
                esperado = len(X_file) - WINDOW_SIZE + 1

                if num_janelas != esperado:
                    raise ValueError(
                        f"{validation_file}: número de janelas inconsistente. "
                        f"Esperado={esperado}, obtido={num_janelas}"
                    )

                print(
                    f"[WINDOW VAL] {os.path.basename(validation_file)}: "
                    f"{num_janelas} janelas"
                )

                X_validation_blocks.extend(blocks)


            X_train_blocks = []

            for train_file, X_file, y_file, preserved_file in train_datasets:
                blocks = create_sliding_window(
                    X_file,
                    WINDOW_SIZE,
                    BATCH_SIZE
                )

                num_janelas = sum(block.shape[0] for block in blocks)
                esperado = len(X_file) - WINDOW_SIZE + 1

                if num_janelas != esperado:
                    raise ValueError(
                        f"{train_file}: número de janelas inconsistente. "
                        f"Esperado={esperado}, obtido={num_janelas}"
                    )

                print(
                    f"[WINDOW] {os.path.basename(train_file)}: "
                    f"{num_janelas} janelas"
                )

                X_train_blocks.extend(blocks)


            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            print(f"Usando dispositivo: {device}")

            # Todas as seções/modelos usam o mesmo conjunto de features
            # dentro desta iteração.
            num_features = train_datasets[0][1].shape[1]

            # Inicializa modelo
            model = AnomalyRNN(
                input_dim=num_features,
                hidden_size=RNN_HIDDEN_SIZE,
                num_layers=RNN_NUM_LAYERS,
                dropout=RNN_DROPOUT
            ).to(device)

            # ----------------- BLOCO DE TREINAMENTO -----------------
            if TRAIN_MODEL:
                print(f"Iniciando treinamento da Seção {secao_id}...")
                optimizer = optim.Adam(model.parameters(), lr=TAXA_APRENDIZADO)
                loss_fn   = nn.MSELoss()
                train_losses = []
                validation_losses = []

                # Variáveis locais para armazenar épocas
                best_validation_loss = float('inf')
                best_epoch = 1
                last_epoch = 0

                if USE_EARLY_STOP:
                    print(f"Usando Early Stopping (paciência={ES_PATIENCE}). "
                          f"Rodando até {MAX_EPOCHS} épocas no máximo.")
                    no_improve_count = 0

                    for epoch in range(MAX_EPOCHS):
                        print(f"(Seção {secao_id}) Epoch {epoch+1} (até {MAX_EPOCHS}, Early Stop ativo)")
                        model.train()

                        train_loss_sum = 0.0
                        train_samples = 0

                        for X_block in X_train_blocks:
                            X_block = X_block.to(device)
                            optimizer.zero_grad()
                            outputs = model(X_block)
                            loss = loss_fn(outputs, X_block)
                            loss.backward()
                            optimizer.step()

                            batch_samples = X_block.size(0)
                            train_loss_sum += loss.item() * batch_samples
                            train_samples += batch_samples

                        if train_samples == 0:
                            raise RuntimeError(
                                "Nenhuma janela disponível para treinamento."
                            )

                        train_epoch_loss = train_loss_sum / train_samples

                        train_losses.append(train_epoch_loss)
                        print(
                            f"(Seção {secao_id}) "
                            f"Perda média de treino: {train_epoch_loss:.6f}"
                        )

                        # Sempre atualiza 'last_epoch'
                        last_epoch = epoch + 1
                        # ============================================================
                        # VALIDAÇÃO DA ÉPOCA
                        # ============================================================
                        if X_validation_blocks is None:
                            raise RuntimeError(
                                "X_validation_blocks ainda não foi criado. "
                                "A divisão treino/validação precisa estar disponível."
                            )

                        model.eval()

                        validation_loss_sum = 0.0
                        validation_samples = 0

                        with torch.no_grad():
                            for X_block in X_validation_blocks:
                                X_block = X_block.to(device)
                                outputs = model(X_block)
                                validation_loss = loss_fn(outputs, X_block)

                                batch_samples = X_block.size(0)
                                validation_loss_sum += validation_loss.item() * batch_samples
                                validation_samples += batch_samples

                        if validation_samples == 0:
                            raise RuntimeError(
                                "Nenhuma janela disponível para validação."
                            )

                        validation_epoch_loss = validation_loss_sum / validation_samples
                        validation_losses.append(validation_epoch_loss)

                        print(
                            f"(Seção {secao_id}) "
                            f"Perda média de validação: {validation_epoch_loss:.6f}"
                        )

                        # Early stopping considera exclusivamente a validação.
                        if validation_epoch_loss < best_validation_loss:
                            best_validation_loss = validation_epoch_loss
                            best_epoch = epoch + 1
                            no_improve_count = 0

                            if SAVE_BEST_MODEL:
                                torch.save(model.state_dict(), model_file)
                                print(
                                    f"(Seção {secao_id}) Novo best model salvo "
                                    f"na época {epoch+1} "
                                    f"(validation loss = {validation_epoch_loss:.6f})"
                                )
                        else:
                            no_improve_count += 1

                        if no_improve_count >= ES_PATIENCE:
                            print(
                                f"EarlyStopping acionado na época {epoch+1}"
                            )
                            break

                else:
                    print(
                        f"Treinando com {num_epochs} épocas fixas "
                        f"(sem Early Stopping)."
                    )

                    for epoch in range(num_epochs):
                        print(
                            f"(Seção {secao_id}) "
                            f"Epoch {epoch+1}/{num_epochs}"
                        )

                        model.train()

                        train_loss_sum = 0.0
                        train_samples = 0


                        for X_block in X_train_blocks:
                            X_block = X_block.to(device)

                            optimizer.zero_grad()
                            outputs = model(X_block)
                            loss = loss_fn(outputs, X_block)

                            loss.backward()
                            optimizer.step()

                            batch_samples = X_block.size(0)
                            train_loss_sum += loss.item() * batch_samples
                            train_samples += batch_samples


                        if train_samples == 0:
                            raise RuntimeError(
                                "Nenhuma janela disponível para treinamento."
                            )

                        train_epoch_loss = train_loss_sum / train_samples
                        train_losses.append(train_epoch_loss)

                        print(
                            f"(Seção {secao_id}) "
                            f"Perda média de treino: "
                            f"{train_epoch_loss:.6f}"
                        )

                        last_epoch = epoch + 1

                        # Validação continua existindo mesmo sem early stopping.
                        if X_validation_blocks is None:
                            raise RuntimeError(
                                "X_validation_blocks ainda não foi criado. "
                                "A divisão treino/validação precisa estar disponível."
                            )

                        model.eval()

                        validation_loss_sum = 0.0
                        validation_samples = 0

                        with torch.no_grad():
                            for X_block in X_validation_blocks:
                                X_block = X_block.to(device)
                                outputs = model(X_block)

                                validation_loss = loss_fn(
                                    outputs,
                                    X_block
                                )

                                batch_samples = X_block.size(0)
                                validation_loss_sum += (
                                    validation_loss.item() * batch_samples
                                )
                                validation_samples += batch_samples

                        if validation_samples == 0:
                            raise RuntimeError(
                                "Nenhuma janela disponível para validação."
                            )

                        validation_epoch_loss = (
                            validation_loss_sum / validation_samples
                        )

                        validation_losses.append(
                            validation_epoch_loss
                        )

                        print(
                            f"(Seção {secao_id}) "
                            f"Perda média de validação: "
                            f"{validation_epoch_loss:.6f}"
                        )

                        # Sem early stopping, mas o melhor checkpoint
                        # continua sendo escolhido pela validação.
                        if (
                            validation_epoch_loss
                            < best_validation_loss
                        ):
                            best_validation_loss = (
                                validation_epoch_loss
                            )
                            best_epoch = epoch + 1

                            if SAVE_BEST_MODEL:
                                torch.save(
                                    model.state_dict(),
                                    model_file
                                )

                # Define qual época foi utilizada de fato para o teste
                if SAVE_BEST_MODEL:
                    epoca_utilizada = best_epoch
                else:
                    epoca_utilizada = last_epoch

                # Cria o gráfico de reconstrução
                plt.figure(figsize=(10,6))

                plt.plot(
                    range(1, len(train_losses) + 1),
                    train_losses,
                    marker='o',
                    label='Treino'
                )

                plt.plot(
                    range(1, len(validation_losses) + 1),
                    validation_losses,
                    marker='o',
                    label='Validação'
                )

                plt.axvline(x=epoca_utilizada, color='red', linestyle='--', label='Época Usada p/ Teste')
                plt.title(f"Perda de Reconstrução - Seção {secao_id}")
                plt.xlabel("Épocas")
                plt.ylabel("MSE")
                plt.legend()
                plt.savefig(os.path.join(pasta_secao, f"perda_reconstrucao_secao{secao_id}.png"))
                plt.close()

                if not SAVE_BEST_MODEL:
                    torch.save(model.state_dict(), model_file)
                    print(f"(Seção {secao_id}) Modelo salvo ao final (última época).")

                # Gera o relatório de treino
                epocas_efetivas = last_epoch
                melhor_perda_validacao = best_validation_loss
                perda_final_treino = train_losses[-1]
                perda_final_validacao = validation_losses[-1]
                melhor_epoca = best_epoch

                relatorio_path = os.path.join(pasta_secao, f"relatorio_treino_secao{secao_id}.txt")
                gera_relatorio_treino_secao(
                    secao_id                = secao_id,
                    epocas_efetivas         = epocas_efetivas,
                    melhor_perda_validacao  = melhor_perda_validacao,
                    perda_final_treino      = perda_final_treino,
                    perda_final_validacao   = perda_final_validacao,
                    melhor_epoca            = melhor_epoca,
                    early_stop              = USE_EARLY_STOP,
                    salvamento_best         = SAVE_BEST_MODEL,
                    output_path             = relatorio_path
                )

            else:
                # Modo TREINO DESATIVADO, carrega modelo pronto
                if os.path.exists(model_file):
                    print(f"Carregando modelo salvo da Seção {secao_id} em {model_file}")
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.eval()
                else:
                    raise FileNotFoundError(f"Modelo da Seção {secao_id} não encontrado em {model_file}")

            # --- Aqui termina o bloco de treinamento de cada seção ---

            # Recarrega o melhor modelo salvo antes do teste
            if TRAIN_MODEL and SAVE_BEST_MODEL:
                print(f"(Seção {secao_id}) Recarregando o melhor modelo salvo para o teste...")
                model.load_state_dict(
                    torch.load(model_file, map_location=device)
                )
                model.eval()

            # ============================================================
            # SCORES NO CONJUNTO NORMAL DE VALIDAÇÃO
            # Estes scores serão usados para calibrar o limiar.
            # ============================================================
            if X_validation_blocks is None:
                raise RuntimeError(
                    "X_validation_blocks ainda não foi criado. "
                    "A divisão treino/validação precisa estar disponível."
                )

            validation_scores = []

            model.eval()
            with torch.no_grad():
                for X_block in X_validation_blocks:
                    X_block = X_block.to(device)
                    outputs = model(X_block)

                    reconstruction_error = torch.mean(
                        (outputs[:, -1, :] - X_block[:, -1, :]) ** 2,
                        dim=1
                    )

                    validation_scores.extend(
                        reconstruction_error.cpu().numpy()
                    )

            validation_scores = np.array(validation_scores)

            if len(validation_scores) == 0:
                raise RuntimeError(
                    "Nenhum score foi produzido para o conjunto de validação."
                )


            # ============================================================
            # DIAGNÓSTICO DOS SCORES NO CONJUNTO DE TREINO
            # Estes scores NÃO são usados para definir o limiar.
            # ============================================================
            train_scores = []

            with torch.no_grad():
                for X_block in X_train_blocks:
                    X_block = X_block.to(device)
                    outputs = model(X_block)

                    reconstruction_error = torch.mean(
                        (outputs[:, -1, :] - X_block[:, -1, :]) ** 2,
                        dim=1
                    )

                    train_scores.extend(
                        reconstruction_error.cpu().numpy()
                    )

            train_scores = np.array(train_scores)

            percentil_train = np.percentile(
                train_scores,
                AUTO_THRESHOLD_PERCENTILE * 100
            )

            max_train = np.max(train_scores)
            min_train = np.min(train_scores)

            print("\n========== DIAGNÓSTICO DO TREINO ==========")
            print(
                f"Percentil {AUTO_THRESHOLD_PERCENTILE * 100:g} "
                f"dos scores de treino (diagnóstico): {percentil_train:.10f}"
            )
            print(f"Menor score de treino: {min_train:.10f}")
            print(f"Maior score de treino: {max_train:.10f}")
            print("OBS.: estes valores NÃO definem o limiar.")
            print("===========================================\n")


            # 2) Define o limiar (fixo ou automático) após ter TODOS os scores
            if USE_AUTO_THRESHOLD:

                # O limiar jamais pode ser calculado a partir do conjunto de teste.
                # validation_scores será produzido pelo conjunto normal de validação,
                # cuja divisão será implementada e auditada separadamente.


                limiar_secao = np.percentile(
                    np.array(validation_scores),
                    AUTO_THRESHOLD_PERCENTILE * 100
                )

                tipo_limiar = (
                    f"Percentil {AUTO_THRESHOLD_PERCENTILE * 100:g} "
                    "da validação normal"
                )


            else:
                raise ValueError(
                    "Limiar fixo não configurado na versão HAI auditada."
                )

            # ============================================================
            # CARREGAMENTO DO TESTE
            #
            # O conjunto de teste somente é carregado após o modelo estar
            # treinado e o limiar ter sido definido pela validação normal.
            # ============================================================
            test_datasets = []

            for test_file in TEST_FILES:
                X_file, y_file, preserved_file = load_hai_file(
                    test_file,
                    features_secao
                )

                test_datasets.append(
                    (test_file, X_file, y_file, preserved_file)
                )

            # ============================================================
            # JANELAMENTO DO TESTE
            # Cada arquivo é processado isoladamente.
            # Nenhuma janela atravessa fronteiras entre arquivos.
            # ============================================================
            X_test_blocks = []

            for test_file, X_file, y_file, preserved_file in test_datasets:
                blocks = create_sliding_window(
                    X_file,
                    WINDOW_SIZE,
                    BATCH_SIZE
                )

                num_janelas = sum(block.shape[0] for block in blocks)
                esperado = len(X_file) - WINDOW_SIZE + 1

                if num_janelas != esperado:
                    raise ValueError(
                        f"{test_file}: número de janelas inconsistente. "
                        f"Esperado={esperado}, obtido={num_janelas}"
                    )

                print(
                    f"[WINDOW TEST] {os.path.basename(test_file)}: "
                    f"{num_janelas} janelas"
                )

                X_test_blocks.extend(blocks)


            # ============ BLOCO DE TESTE =============
            print(f"(Seção {secao_id}) Processando blocos de teste...")
            model.eval()

            # ============================================================
            # ALINHAMENTO DOS TESTES
            #
            # Cada arquivo perde apenas suas próprias WINDOW_SIZE - 1
            # primeiras amostras. Assim, o alinhamento respeita as
            # fronteiras entre test1, test2, test3 e test4.
            # ============================================================

            y_true_partes = []
            preserved_partes = []
            X_test_partes = []

            for test_file, X_file, y_file, preserved_file in test_datasets:

                esperado = len(X_file) - WINDOW_SIZE + 1

                if esperado <= 0:
                    raise ValueError(
                        f"{test_file}: arquivo menor que WINDOW_SIZE."
                    )

                y_parte = (
                    y_file
                    .iloc[WINDOW_SIZE - 1:]
                    .to_numpy()
                )

                preserved_parte = (
                    preserved_file
                    .iloc[WINDOW_SIZE - 1:]
                    .reset_index(drop=True)
                )

                X_parte = (
                    X_file
                    .iloc[WINDOW_SIZE - 1:]
                    .reset_index(drop=True)
                )

                if not (
                    len(y_parte)
                    == len(preserved_parte)
                    == len(X_parte)
                    == esperado
                ):
                    raise ValueError(
                        f"{test_file}: erro de alinhamento após janelamento."
                    )

                print(
                    f"[ALIGN] {os.path.basename(test_file)}: "
                    f"{esperado} amostras avaliáveis"
                )

                preserved_parte["arquivo_origem"] = os.path.basename(test_file)
                y_true_partes.append(y_parte)
                preserved_partes.append(preserved_parte)
                X_test_partes.append(X_parte)

            # Concatenação acontece SOMENTE depois do alinhamento individual.
            # Nenhuma janela atravessa a fronteira entre arquivos.
            y_true = np.concatenate(y_true_partes)

            preserved_test_data = pd.concat(
                preserved_partes,
                ignore_index=True
            )

            X_test_alinhado = pd.concat(
                X_test_partes,
                ignore_index=True
            )

            if not (
                len(y_true)
                == len(preserved_test_data)
                == len(X_test_alinhado)
            ):
                raise ValueError(
                    "Desalinhamento global após combinar os quatro testes."
                )

            print(
                f"[ALIGN] Total de amostras avaliáveis no teste: "
                f"{len(y_true)}"
            )

            # ============================================================
            # AVALIAÇÃO NO CONJUNTO DE TESTE
            # O limiar já está definido antes de calcular os scores de teste.
            # ============================================================
            anomaly_scores = []

            model.eval()
            with torch.no_grad():
                for X_block in X_test_blocks:
                    X_block = X_block.to(device)
                    outputs = model(X_block)

                    reconstruction_error = torch.mean(
                        (outputs[:, -1, :] - X_block[:, -1, :]) ** 2,
                        dim=1
                    )

                    anomaly_scores.extend(
                        reconstruction_error.cpu().numpy()
                    )

            # Gera y_pred para TODO o conjunto de teste
            y_pred_full = (np.array(anomaly_scores) > limiar_secao).astype(int)

            # ============================================================
            # VERIFICAÇÃO ESTRITA DE ALINHAMENTO
            # ============================================================
            if len(y_pred_full) != len(y_true):
                raise ValueError(
                    "Desalinhamento entre scores/predições e rótulos: "
                    f"predições={len(y_pred_full)}, "
                    f"rótulos={len(y_true)}"
                )

            if len(y_true) != len(preserved_test_data):
                raise ValueError(
                    "Desalinhamento entre rótulos e dados preservados: "
                    f"rótulos={len(y_true)}, "
                    f"preservados={len(preserved_test_data)}"
                )

            if len(y_true) != len(X_test_alinhado):
                raise ValueError(
                    "Desalinhamento entre rótulos e features de teste: "
                    f"rótulos={len(y_true)}, "
                    f"features={len(X_test_alinhado)}"
                )

            y_pred = y_pred_full

            # ==========================================
            # Fusão com df_unificado (para saída global)
            # ==========================================
            if df_unificado is None:
                # PRIMEIRA SEÇÃO => Adiciona Rótulo_Real
                df_temp = pd.DataFrame({
                    "timestamp": preserved_test_data["timestamp"].values,
                    "arquivo_origem": preserved_test_data["arquivo_origem"].values,
                    "Tipo": preserved_test_data["Tipo"].values,
                    "Rótulo_Real": y_true,
                    f"Secao_{secao_id}": y_pred
                })
            else:
                # Demais seções => NÃO adiciona Rótulo_Real
                df_temp = pd.DataFrame({
                    "timestamp": preserved_test_data["timestamp"].values,
                    "arquivo_origem": preserved_test_data["arquivo_origem"].values,
                    "Tipo": preserved_test_data["Tipo"].values,
                    f"Secao_{secao_id}": y_pred
                })

            saida_secao_file = os.path.join(pasta_secao, f"saida_rnn_secao{secao_id}.csv")
            df_temp.to_csv(saida_secao_file, index=False, encoding="utf-8")
            print(f"(Seção {secao_id}) Arquivo de detecção gerado: {saida_secao_file}")

            # ============================================================
            # SAÍDA DE SCORES CONTÍNUOS
            # Arquivo independente para futuras estratégias de fusão.
            # ============================================================
            df_scores = pd.DataFrame({
                "timestamp": preserved_test_data["timestamp"].values,
                "arquivo_origem": preserved_test_data["arquivo_origem"].values,
                "Tipo": preserved_test_data["Tipo"].values,
                "Rótulo_Real": y_true,
                "Score_Anomalia": np.array(anomaly_scores),
                "Limiar": np.full(len(y_true), limiar_secao),
                "Rótulo_Previsto": y_pred
            })

            saida_scores_file = os.path.join(
                pasta_secao,
                f"scores_rnn_secao{secao_id}.csv"
            )

            df_scores.to_csv(
                saida_scores_file,
                index=False,
                encoding="utf-8"
            )

            print(
                f"(Seção {secao_id}) Arquivo de scores contínuos gerado: "
                f"{saida_scores_file}"
            )

            # Criar relatório de limiar
            relatorio_limiar_path = os.path.join(pasta_secao, f"relatorio_limiar_secao{secao_id}.txt")
            gerar_relatorio_limiar(secao_id, limiar_secao, tipo_limiar, relatorio_limiar_path)
            relatorios_limiar.append(relatorio_limiar_path)

            # Atualiza df_unificado
            if df_unificado is None:
                df_unificado = df_temp
            else:
                df_unificado = pd.merge(
                    df_unificado,
                    df_temp,
                    on=["timestamp", "arquivo_origem", "Tipo"],
                    how="outer"
)

            # Ajusta subX_test e subPreserved
            subX_test = X_test_alinhado
            subPreserved = preserved_test_data

            # Cria dataset_resultado_completo.csv local
            _ = create_consolidated_dataset(
                X_test=subX_test,
                y_test=y_true,
                y_pred=y_pred,
                preserved_data=subPreserved,
                output_folder=pasta_secao
            )

            # Gera métricas e mapas para essa seção
            generate_test_results(
                y_true = y_true,
                y_pred = y_pred,
                test_df = subX_test,
                anomaly_scores = np.array(anomaly_scores),
                threshold = limiar_secao,
                output_folder = pasta_secao
            )
            # ============================================================
            # LIMPEZA DE MEMÓRIA ENTRE SEÇÕES
            # ============================================================
            print(f"(Seção {secao_id}) Iniciando limpeza de memória...")

            plt.close('all')

            del model

            del train_datasets
            del test_datasets

            del X_train_blocks
            del X_test_blocks

            del X_test_alinhado
            del y_true_partes
            del preserved_partes
            del X_test_partes

            del anomaly_scores
            del train_scores

            del preserved_test_data
            del subX_test
            del subPreserved

            del y_pred
            del y_true

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()



        # (Após rodar todas as seções)
        pasta_geral = os.path.join(RESULTADO_BASE, "geral")
        os.makedirs(pasta_geral, exist_ok=True)
        final_path = os.path.join(pasta_geral, "saida_final_unificado.csv")

        df_unificado.sort_values(
            by=["arquivo_origem", "timestamp"],
            inplace=True
        )
        df_unificado.to_csv(final_path, index=False, encoding='utf-8')
        print(f"Arquivo final unificado salvo em: {final_path}")

        # Gera o relatório geral apenas com as seções da rodada atual
        gera_relatorio_geral(RESULTADO_BASE, MODELOS_ATIVOS)

        # Cria cópia para gerar métricas gerais
        df_geral = df_unificado.copy()

        # 2) Identifica as colunas de seções (colunas que começam com 'Secao_')
        secao_cols = [c for c in df_geral.columns if c.startswith("Secao_")]

        # 3) Cria 'Rótulo_Previsto_Geral' = 1 se qualquer Secao_i == 1
        if not secao_cols:
            raise ValueError("Não há colunas 'Secao_' no df_unificado; impossível gerar rótulo geral.")
        df_geral["Rótulo_Previsto"] = df_geral[secao_cols].max(axis=1)

        # 4) Verifica se 'Rótulo_Real' existe
        if "Rótulo_Real" not in df_geral.columns:
            raise ValueError("Coluna 'Rótulo_Real' não presente no CSV unificado.")

        # Salva de volta no CSV final
        df_geral.to_csv(final_path, index=False, encoding='utf-8')

        # Criar relatório global de limiares
        relatorio_global_limiar_path = os.path.join(pasta_geral, "relatorio_limiar_geral.txt")
        atualizar_relatorio_global_limiar(relatorio_global_limiar_path, relatorios_limiar)
        print(f"Relatório global de limiares salvo em: {relatorio_global_limiar_path}")

        # Gera a Matriz de Confusão, métricas e mapa de ataques gerais
        y_true_geral = df_geral["Rótulo_Real"].values
        y_pred_geral = df_geral["Rótulo_Previsto"].values

        df_final_geral = df_geral[
            [
                "timestamp",
                "arquivo_origem",
                "Tipo",
                "Rótulo_Real",
                "Rótulo_Previsto"
            ]
        ].copy()
        df_final_geral_path = os.path.join(pasta_geral, "dataset_resultado_completo.csv")
        df_final_geral.to_csv(df_final_geral_path, index=False, encoding="utf-8")

        create_and_save_confusion_matrix(y_true_geral, y_pred_geral, pasta_geral)
        calculate_and_save_metrics(
            y_true_geral,
            y_pred_geral,
            None,
            pasta_geral
        )
        map_real_vs_predicted_attacks(pasta_geral)

        # Finaliza com sucesso
        print("Execução concluída com sucesso.")

    except Exception as e:
        print(f"Erro durante a execução: {e}")
        raise


if __name__ == "__main__":
    main()
