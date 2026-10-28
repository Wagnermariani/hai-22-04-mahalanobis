import os

import numpy as np
import pandas as pd
import torch

from rnn_hai_auditada import (
    AnomalyRNN,
    load_hai_file,
    create_sliding_window,
    WINDOW_SIZE,
    BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_NUM_LAYERS,
        RNN_DROPOUT,
    TRAIN_FILES,
    VALIDATION_FILES,
    TEST_FILES,
    DIRETORIO_PROJETO,
)



# ============================================================================
# >>>>>> CONFIGURAÇÃO <<<<<<
# ============================================================================

CHECKPOINT = os.path.join(
    DIRETORIO_PROJETO,
    "resultado",
    "rnn_hai",
    "secaoMONOLITICO",
    "modelo_treinado_rnn_secaoMONOLITICO.pth"
)

ARQUIVO_SCORES_TESTE = os.path.join(
    DIRETORIO_PROJETO,
    "resultado",
    "rnn_hai",
    "secaoMONOLITICO",
    "scores_rnn_secaoMONOLITICO.csv"
)

DIRETORIO_SAIDA = os.path.join(
    DIRETORIO_PROJETO,
    "resultado",
    "rnn_hai",
    "secaoMONOLITICO",
    "residuos_monolitico"
)

ARQUIVO_RESIDUOS_TREINO = os.path.join(
    DIRETORIO_SAIDA,
    "residuos_treino_monolitico.csv"
)

ARQUIVO_RESIDUOS_VALIDACAO = os.path.join(
    DIRETORIO_SAIDA,
    "residuos_validacao_monolitico.csv"
)

ARQUIVO_RESIDUOS_TESTE = os.path.join(
    DIRETORIO_SAIDA,
    "residuos_teste_monolitico.csv"
)

os.makedirs(DIRETORIO_SAIDA, exist_ok=True)


# ============================================================================
# >>>>>> EXTRAÇÃO DOS RESÍDUOS <<<<<<
# ============================================================================

def extrair_residuos(
    arquivos,
    features,
    model,
    device,
    nome_conjunto
):
    partes = []

    model.eval()

    for arquivo in arquivos:
        X_file, y_file, preserved_file = load_hai_file(
            arquivo,
            features
        )

        blocks = create_sliding_window(
            X_file,
            WINDOW_SIZE,
            BATCH_SIZE
        )

        esperado = len(X_file) - WINDOW_SIZE + 1
        residuos_partes = []
        erros_assinados_partes = []

        with torch.no_grad():
            for X_block in blocks:
                X_block = X_block.to(device)

                outputs = model(X_block)

                erros_assinados = (
                    outputs[:, -1, :] - X_block[:, -1, :]
                )

                residuos = erros_assinados ** 2

                residuos_partes.append(
                    residuos.cpu().numpy()
                )

                erros_assinados_partes.append(
                    erros_assinados.cpu().numpy()
                )

        residuos_file = np.concatenate(
            residuos_partes,
            axis=0
        )

        erros_assinados_file = np.concatenate(
            erros_assinados_partes,
            axis=0
        )

        if erros_assinados_file.shape != (esperado, len(features)):
            raise ValueError(
                f"{arquivo}: dimensão inesperada dos erros assinados. "
                f"Esperado=({esperado}, {len(features)}), "
                f"obtido={erros_assinados_file.shape}"
            )

        if residuos_file.shape != (esperado, len(features)):
            raise ValueError(
                f"{arquivo}: dimensão inesperada dos resíduos. "
                f"Esperado=({esperado}, {len(features)}), "
                f"obtido={residuos_file.shape}"
            )

        metadata = (
            preserved_file
            .iloc[WINDOW_SIZE - 1:]
            .reset_index(drop=True)
            .copy()
        )

        if len(metadata) != esperado:
            raise ValueError(
                f"{arquivo}: desalinhamento dos metadados."
            )

        df_residuos = pd.DataFrame(
            residuos_file,
            columns=[
                f"res_{feature}"
                for feature in features
            ]
        )

        df_erros_assinados = pd.DataFrame(
            erros_assinados_file,
            columns=[
                f"err_{feature}"
                for feature in features
            ]
        )

        score_global = residuos_file.mean(axis=1)

        resultado = pd.concat(
            [
                metadata[
                    ["timestamp", "Tipo", "Attack"]
                ],
                df_residuos,
                df_erros_assinados
            ],
            axis=1
        )

        resultado.insert(
            1,
            "arquivo_origem",
            os.path.basename(arquivo)
        )

        resultado["Score_Global"] = score_global

        partes.append(resultado)

        print(
            f"[RESIDUOS {nome_conjunto}] "
            f"{os.path.basename(arquivo)}: "
            f"{len(resultado)} linhas"
        )

    resultado_final = pd.concat(
        partes,
        ignore_index=True
    )

    return resultado_final


# ============================================================================
# >>>>>> PROGRAMA PRINCIPAL <<<<<<
# ============================================================================

def main():

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {CHECKPOINT}"
        )

    # A ordem das features deve ser exatamente a mesma usada pela RNN.
    colunas = pd.read_csv(
        VALIDATION_FILES[0],
        nrows=0,
        encoding="utf-8"
    ).columns.str.strip()

    features = [
        coluna
        for coluna in colunas
        if coluna not in ["timestamp", "Attack", "Tipo"]
    ]

    if len(features) != 86:
        raise ValueError(
            f"Esperadas 86 features, encontradas {len(features)}."
        )

    print(f"Features utilizadas: {len(features)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Dispositivo: {device}")

    model = AnomalyRNN(
        input_dim=len(features),
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        dropout=RNN_DROPOUT
    ).to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device
        )
    )

    model.eval()

    print("Checkpoint carregado com sucesso.")

    # ------------------------------------------------------------------------
    # VALIDAÇÃO
    # ------------------------------------------------------------------------

    residuos_validacao = extrair_residuos(
        VALIDATION_FILES,
        features,
        model,
        device,
        "VALIDACAO"
    )

    residuos_validacao.to_csv(
        ARQUIVO_RESIDUOS_VALIDACAO,
        index=False
    )

    print(
        f"Resíduos da validação salvos em: "
        f"{ARQUIVO_RESIDUOS_VALIDACAO}"
    )

    p99_validacao = np.percentile(
        residuos_validacao["Score_Global"].to_numpy(),
        99
    )

    print(
        f"P99 recalculado da validação: "
        f"{p99_validacao:.10f}"
    )

    # ------------------------------------------------------------------------
    # TESTE
    # ------------------------------------------------------------------------

    residuos_teste = extrair_residuos(
        TEST_FILES,
        features,
        model,
        device,
        "TESTE"
    )

    # ------------------------------------------------------------------------
    # AUDITORIA CONTRA O SCORE GLOBAL ORIGINAL
    # ------------------------------------------------------------------------

    scores_originais = pd.read_csv(
        ARQUIVO_SCORES_TESTE
    )

    if len(scores_originais) != len(residuos_teste):
        raise ValueError(
            "Quantidade de scores diferente da execução original: "
            f"original={len(scores_originais)}, "
            f"recalculado={len(residuos_teste)}"
        )

    if not np.array_equal(
        scores_originais["timestamp"].astype(str).to_numpy(),
        residuos_teste["timestamp"].astype(str).to_numpy()
    ):
        raise ValueError(
            "Timestamps não coincidem com a execução original."
        )

    if not np.array_equal(
        scores_originais["arquivo_origem"].astype(str).to_numpy(),
        residuos_teste["arquivo_origem"].astype(str).to_numpy()
    ):
        raise ValueError(
            "Arquivos de origem não coincidem com a execução original."
        )

    score_original = (
        scores_originais["Score_Anomalia"]
        .to_numpy(dtype=np.float64)
    )

    score_recalculado = (
        residuos_teste["Score_Global"]
        .to_numpy(dtype=np.float64)
    )

    diferenca_maxima = np.max(
        np.abs(
            score_original - score_recalculado
        )
    )

    print(
        f"Maior diferença absoluta entre score original "
        f"e score recalculado: {diferenca_maxima:.12e}"
    )

    if not np.allclose(
        score_original,
        score_recalculado,
        rtol=1e-6,
        atol=1e-8
    ):
        raise ValueError(
            "AUDITORIA FALHOU: o score reconstruído a partir "
            "dos 86 resíduos não coincide com o score original."
        )

    print(
        "AUDITORIA OK: a média dos 86 resíduos reproduz "
        "o Score_Anomalia original."
    )

    residuos_teste.to_csv(
        ARQUIVO_RESIDUOS_TESTE,
        index=False
    )

    print(
        f"Resíduos do teste salvos em: "
        f"{ARQUIVO_RESIDUOS_TESTE}"
    )

    print("\nExtração concluída com sucesso.")


if __name__ == "__main__":
    main()