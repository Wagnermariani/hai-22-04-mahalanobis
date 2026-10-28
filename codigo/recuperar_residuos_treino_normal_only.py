#!/usr/bin/env python3
"""
HAI 22.04 - Recuperacao dos residuos normais de treino.

Reutiliza a RNN monolitica congelada.
Nao treina modelos.
Nao consulta o conjunto de teste.
Nao modifica arquivos historicos.
Nao sobrescreve resultados existentes.
"""

import os
import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from rnn_hai_auditada import (
    AnomalyRNN,
    load_hai_file,
    create_sliding_window,
    TRAIN_FILES,
    WINDOW_SIZE,
    BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_NUM_LAYERS,
    RNN_DROPOUT,
)

BASE = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    BASE / "resultado/rnn_hai/secaoMONOLITICO/"
    "modelo_treinado_rnn_secaoMONOLITICO.pth"
)

REFERENCIA = (
    BASE / "resultado/rnn_hai/secaoMONOLITICO/"
    "mahalanobis_global/parametros_mahalanobis_global.npz"
)

DESTINO = (
    BASE / "resultado/"
    "diagnostico_mahalanobis_normal_only"
)

SAIDA = DESTINO / "residuos_treino_assinados.npy"

SHAPE_ESPERADO = (805692, 86)


def extrair_residuos(arquivos, features, model, device):
    partes = []
    model.eval()

    for caminho in arquivos:
        X_file, y, _ = load_hai_file(caminho, features)

        if not (y.to_numpy() == 0).all():
            raise RuntimeError(
                f"Arquivo de treino contem ataques: {caminho}"
            )

        blocks = create_sliding_window(
            X_file, WINDOW_SIZE, BATCH_SIZE
        )

        esperado = len(X_file) - WINDOW_SIZE + 1
        residuos_partes = []

        with torch.no_grad():
            for X_block in blocks:
                X_block = X_block.to(device)
                outputs = model(X_block)

                # Mesma convencao do programa original.
                erros = (
                    outputs[:, -1, :]
                    - X_block[:, -1, :]
                )

                residuos_partes.append(
                    erros.detach().cpu().numpy().astype(
                        np.float32, copy=False
                    )
                )

        residuos_file = np.concatenate(
            residuos_partes, axis=0
        )

        if residuos_file.shape != (
            esperado, len(features)
        ):
            raise RuntimeError(
                f"Shape inesperado em {caminho}: "
                f"{residuos_file.shape}"
            )

        print(
            f"[TREINO] {os.path.basename(caminho)}: "
            f"{len(residuos_file)} residuos",
            flush=True
        )

        partes.append(residuos_file)

        del X_file, blocks, residuos_partes
        gc.collect()

    return np.concatenate(partes, axis=0)


def comparar_com_historico(residuos):
    print("\n=== REPRODUTIBILIDADE GLOBAL ===", flush=True)

    modelo = LedoitWolf(assume_centered=False)
    modelo.fit(residuos)

    comparacoes = [
        ("media", modelo.location_),
        ("covariancia", modelo.covariance_),
        ("precisao", modelo.precision_),
        ("shrinkage", np.array([modelo.shrinkage_])),
    ]

    todas_equivalentes = True

    with np.load(REFERENCIA, allow_pickle=False) as antigo:
        for nome, atual in comparacoes:
            referencia = antigo[nome]

            if atual.shape != referencia.shape:
                raise RuntimeError(
                    f"Shape diferente em {nome}"
                )

            diferenca = float(
                np.max(np.abs(atual - referencia))
            )

            equivalente = bool(
                np.allclose(
                    atual,
                    referencia,
                    rtol=1e-4,
                    atol=1e-6
                )
            )

            todas_equivalentes &= equivalente

            print(
                f"{nome}: "
                f"erro_abs_max={diferenca:.12e}; "
                f"allclose={equivalente}",
                flush=True
            )

    return todas_equivalentes


def salvar_sem_sobrescrever(residuos):
    if SAIDA.exists() or SAIDA.is_symlink():
        raise FileExistsError(
            f"Arquivo existente: {SAIDA}"
        )

    DESTINO.mkdir(parents=True, exist_ok=True)

    # O arquivo temporario fica no proprio destino.
    # Somente apos terminar a gravacao ele e publicado.
    temporario = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npy.tmp",
            prefix=".residuos_",
            dir=DESTINO,
            delete=False
        ) as arq:
            temporario = Path(arq.name)
            np.save(arq, residuos)
            arq.flush()
            os.fsync(arq.fileno())

        # Link atomico: falha se o destino ja existir.
        os.link(temporario, SAIDA)

    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)

    print(f"\nArquivo novo salvo: {SAIDA}")


def main():
    print("=== HAI: RECUPERACAO NORMAL-ONLY ===")

    if SAIDA.exists() or SAIDA.is_symlink():
        raise FileExistsError(
            f"Saida ja existe; nenhuma sobrescrita: {SAIDA}"
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if not REFERENCIA.is_file():
        raise FileNotFoundError(REFERENCIA)

    colunas = pd.read_csv(
        TRAIN_FILES[0],
        nrows=0,
        encoding="utf-8"
    ).columns.str.strip()

    features = [
        c for c in colunas
        if c not in ("timestamp", "Attack", "Tipo")
    ]

    if len(features) != 86 or len(set(features)) != 86:
        raise RuntimeError(
            "Lista de features invalida."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo:", device, flush=True)

    model = AnomalyRNN(
        input_dim=86,
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        dropout=RNN_DROPOUT
    ).to(device)

    pesos = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(pesos)
    model.eval()

    residuos = extrair_residuos(
        TRAIN_FILES, features, model, device
    )

    if residuos.shape != SHAPE_ESPERADO:
        raise RuntimeError(
            f"Shape inesperado: {residuos.shape}"
        )

    if residuos.dtype != np.float32:
        raise RuntimeError(
            f"Dtype inesperado: {residuos.dtype}"
        )

    if not np.isfinite(residuos).all():
        raise RuntimeError(
            "Residuos contem NaN ou Inf."
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    equivalente = comparar_com_historico(residuos)

    if not equivalente:
        print(
            "\nATENCAO: reproducao numerica ainda "
            "nao validada. Nenhum residuo foi salvo."
        )
        raise SystemExit(2)

    salvar_sem_sobrescrever(residuos)

    print(
        "\nReproducao numerica aprovada "
        "pelas tolerancias preliminares."
    )


if __name__ == "__main__":
    main()
