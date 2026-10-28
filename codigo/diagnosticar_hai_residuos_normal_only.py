
#!/usr/bin/env python3
"""
Diagnostico geometrico normal-only - HAI 22.04.

Nao executa RNN.
Nao utiliza dados de teste ou ataques.
Nao modifica resultados historicos.
Nao sobrescreve resultados anteriores.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import eigvalsh
from sklearn.covariance import LedoitWolf


BASE = Path(__file__).resolve().parent.parent

TREINO = (
    BASE / "resultado/diagnostico_mahalanobis_normal_only/"
    "residuos_treino_assinados.npy"
)

VALIDACAO = (
    BASE / "resultado/rnn_hai/secaoMONOLITICO/"
    "residuos_monolitico/residuos_validacao_monolitico.csv"
)

CABECALHO = (
    BASE / "dataset_trabalho/hai-22.04/"
    "normalizados_oficiais/treino/"
    "train1_treino_normalizado.csv"
)

REFERENCIA = (
    BASE / "resultado/rnn_hai/secaoMONOLITICO/"
    "mahalanobis_global/parametros_mahalanobis_global.npz"
)

DESTINO = (
    BASE / "resultado/diagnostico_mahalanobis_normal_only/"
    "diagnostico"
)

FPR_ALVO = 0.01

TAMANHOS = {
    "P1": 44,
    "P2": 24,
    "P3": 7,
    "P4": 11,
}


def main():

    print("=== DIAGNOSTICO NORMAL-ONLY HAI ===", flush=True)

    # Nunca reutilizar um diretorio de resultados.
    if DESTINO.exists() or DESTINO.is_symlink():
        raise RuntimeError(
            f"Destino ja existe: {DESTINO}"
        )

    # Ordem oficial das variaveis.
    cabecalho = pd.read_csv(
        CABECALHO, nrows=0
    ).columns.str.strip().tolist()

    features = [
        c for c in cabecalho
        if c not in ("timestamp", "Attack", "Tipo")
    ]

    if len(features) != 86 or len(set(features)) != 86:
        raise RuntimeError("Lista de features invalida.")

    grupos = {}
    indices = {}

    for grupo, tamanho in TAMANHOS.items():
        idx = np.array(
            [
                i for i, feature in enumerate(features)
                if feature.startswith(f"{grupo}_")
            ],
            dtype=np.int64
        )

        if len(idx) != tamanho:
            raise RuntimeError(
                f"{grupo}: esperado={tamanho}, obtido={len(idx)}"
            )

        grupos[grupo] = [
            features[i] for i in idx
        ]
        indices[grupo] = idx

    todos = np.concatenate(list(indices.values()))

    if not np.array_equal(
        np.sort(todos), np.arange(86)
    ):
        raise RuntimeError(
            "Grupos sobrepostos ou variaveis ausentes."
        )

    # Residuos normais de treinamento recuperados.
    X_train = np.load(
        TREINO, allow_pickle=False
    )

    if (
        X_train.shape != (805692, 86)
        or X_train.dtype != np.float32
        or not np.isfinite(X_train).all()
    ):
        raise RuntimeError(
            "Residuos de treino invalidos."
        )

    # Validacao normal, na mesma ordem do treinamento.
    colunas_err = [f"err_{c}" for c in features]

    validacao = pd.read_csv(
        VALIDACAO,
        usecols=colunas_err + ["Attack"],
        dtype={c: np.float32 for c in colunas_err}
    )

    if list(validacao[colunas_err].columns) != colunas_err:
        raise RuntimeError(
            "Ordem das features divergente."
        )

    if not (
        pd.to_numeric(
            validacao["Attack"], errors="raise"
        ) == 0
    ).all():
        raise RuntimeError(
            "Validacao contem registros de ataque."
        )

    X_val = validacao[colunas_err].to_numpy(
        dtype=np.float32, copy=False
    )

    if (
        X_val.shape != (197646, 86)
        or not np.isfinite(X_val).all()
    ):
        raise RuntimeError(
            "Residuos de validacao invalidos."
        )

    # Residuos preservados no arquivo original em float32.
    # Ajustes estatisticos realizados em dupla precisao.
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)

    print(
        f"Treino: {X_train.shape}; "
        f"validacao: {X_val.shape}; "
        f"precisao estatistica: float64",
        flush=True
    )

    # Mahalanobis global em dupla precisao.
    global_model = LedoitWolf(
        store_precision=True,
        assume_centered=False
    )
    global_model.fit(X_train)

    # O modelo historico foi ajustado em float32.
    # Sua reproducao exata ja foi comprovada separadamente.
    # Aqui verificamos a estabilidade do novo ajuste float64.
    C = global_model.covariance_
    P = global_model.precision_

    erro_inversa = np.linalg.norm(
        C @ P - np.eye(len(features)),
        ord=np.inf
    )

    menor_autovalor = float(
        np.linalg.eigvalsh(
            (P + P.T) / 2.0
        )[0]
    )

    print(
        f"Global float64: erro_inversa={erro_inversa:.6e}; "
        f"menor_autovalor={menor_autovalor:.6e}",
        flush=True
    )

    if (
        not np.isfinite(erro_inversa)
        or erro_inversa > 1e-6
        or menor_autovalor <= 0
    ):
        raise RuntimeError(
            "Mahalanobis global float64 numericamente instavel."
        )

    score_global_val = global_model.mahalanobis(X_val)

    tau_global = float(
        np.quantile(
            score_global_val,
            1.0 - FPR_ALVO
        )
    )

    media_global = float(np.mean(score_global_val))
    gap_global = tau_global - media_global

    if gap_global <= 0:
        raise RuntimeError("Gap global nao positivo.")

    # Mahalanobis por processo industrial.
    modelos_locais = {}
    scores_locais = {}

    for grupo, idx in indices.items():
        print(
            f"Ajustando Mahalanobis local: {grupo}",
            flush=True
        )

        modelo = LedoitWolf(
            store_precision=True,
            assume_centered=False
        )

        modelo.fit(X_train[:, idx])

        modelos_locais[grupo] = modelo
        scores_locais[grupo] = modelo.mahalanobis(
            X_val[:, idx]
        )

    matriz_locais = np.column_stack(
        [
            scores_locais[grupo]
            for grupo in TAMANHOS
        ]
    )

    score_local_max = np.max(
        matriz_locais, axis=1
    )

    tau_local = float(
        np.quantile(
            score_local_max,
            1.0 - FPR_ALVO
        )
    )

    n_alarmes_global = int(
        np.sum(score_global_val > tau_global)
    )
    n_alarmes_local = int(
        np.sum(score_local_max > tau_local)
    )

    resultados = []

    for grupo, idx in indices.items():

        media_local = float(
            np.mean(scores_locais[grupo])
        )

        gap_local = tau_local - media_local

        if gap_local <= 0:
            raise RuntimeError(
                f"{grupo}: gap local nao positivo."
            )

        burden = gap_global / gap_local

        Pg = global_model.precision_[
            np.ix_(idx, idx)
        ]

        Pl = modelos_locais[grupo].precision_

        Pg = (Pg + Pg.T) / 2.0
        Pl = (Pl + Pl.T) / 2.0

        autovalores = eigvalsh(Pg, Pl)

        if (
            not np.isfinite(autovalores).all()
            or np.any(autovalores <= 0)
        ):
            raise RuntimeError(
                f"{grupo}: autovalores invalidos."
            )

        r_min = float(np.min(autovalores))
        r_med = float(np.median(autovalores))
        r_max = float(np.max(autovalores))

        resultados.append({
            "Grupo": grupo,
            "Tamanho": len(idx),
            "Tau_Global": tau_global,
            "Media_Global": media_global,
            "Gap_Global": gap_global,
            "Tau_Local_Fusao": tau_local,
            "Media_Local": media_local,
            "Gap_Local": gap_local,
            "Burden_Ratio": burden,
            "R_Min": r_min,
            "R_Mediano": r_med,
            "R_Max": r_max,
            "Fator_Pessimista": burden / r_max,
            "Fator_Mediano": burden / r_med,
            "Fator_Otimista": burden / r_min,
        })

    df = pd.DataFrame(resultados)

    fatores = df[
        "Fator_Pessimista"
    ].to_numpy()

    burdens = df[
        "Burden_Ratio"
    ].to_numpy()

    resumo = pd.DataFrame([{
        "Dataset": "HAI_22.04",
        "Espaco": "Residuos_RNN",
        "Precisao_Estatistica": "float64",
        "N_Grupos": len(TAMANHOS),
        "FPR_Alvo": FPR_ALVO,
        "Tau_Global": tau_global,
        "Tau_Local": tau_local,
        "Gap_Global": gap_global,
        "Alarmes_Val_Global": n_alarmes_global,
        "Alarmes_Val_Local": n_alarmes_local,
        "Burden_Min": float(burdens.min()),
        "Burden_Mediana": float(np.median(burdens)),
        "Burden_Max": float(burdens.max()),
        "Fator_Pessimista_Min": float(fatores.min()),
        "Fator_Pessimista_Mediana": float(
            np.median(fatores)
        ),
        "Fator_Pessimista_Max": float(fatores.max()),
        "Grupos_Fator_Pessimista_Maior_1": int(
            np.sum(fatores > 1.0)
        ),
    }])

    # Todos os calculos terminaram.
    # Criacao exclusiva: falha se o destino ja existir.
    DESTINO.mkdir(parents=False, exist_ok=False)

    detalhe = DESTINO / "diagnostico_por_grupo.csv"
    arquivo_resumo = DESTINO / "diagnostico_resumo.csv"

    with detalhe.open("x", encoding="utf-8") as f:
        df.to_csv(f, index=False)

    with arquivo_resumo.open("x", encoding="utf-8") as f:
        resumo.to_csv(f, index=False)

    print("\n=== FATORES POR GRUPO ===")
    print(df[
        [
            "Grupo",
            "Tamanho",
            "Burden_Ratio",
            "R_Min",
            "R_Max",
            "Fator_Pessimista",
            "Fator_Mediano",
            "Fator_Otimista",
        ]
    ].to_string(index=False))

    print("\n=== RESUMO ===")
    print(resumo.to_string(index=False))

    print("\nArquivos novos:")
    print(detalhe)
    print(arquivo_resumo)
    print("\nNenhum dado de teste foi consultado.")


if __name__ == "__main__":
    main()
