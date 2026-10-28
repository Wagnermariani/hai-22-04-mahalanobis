import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "dataset_trabalho" / "hai-22.04"
SAIDA = BASE / "rotulados"

SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVOS_ATIVOS = [
    "train1.csv",
    "train2.csv",
    "train3.csv",
    "train4.csv",
    "train5.csv",
    "train6.csv",
    "test1.csv",
    "test2.csv",
    "test3.csv",
    "test4.csv",
]

CONFIG_TESTES = {
    "test1.csv": {"prefixo": 1, "eventos": 7,  "ataques": 885},
    "test2.csv": {"prefixo": 2, "eventos": 17, "ataques": 2881},
    "test3.csv": {"prefixo": 3, "eventos": 10, "ataques": 3841},
    "test4.csv": {"prefixo": 4, "eventos": 24, "ataques": 4423},
}


def rotular_arquivo(nome_arquivo):

    print("\n" + "=" * 90)
    print(f"PROCESSANDO: {nome_arquivo}")
    print("=" * 90)

    entrada = BASE / nome_arquivo

    if not entrada.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {entrada}")

    df = pd.read_csv(entrada)
    original = df.copy(deep=True)

    colunas_originais = df.columns.tolist()

    if "Attack" not in df.columns:
        raise RuntimeError(f"{nome_arquivo}: coluna Attack não encontrada.")

    if "Tipo" in df.columns:
        raise RuntimeError(f"{nome_arquivo}: cópia-base já possui coluna Tipo.")

    # ========================================================
    # TREINAMENTO
    # ========================================================

    if nome_arquivo.startswith("train"):

        if int((df["Attack"] != 0).sum()) != 0:
            raise RuntimeError(
                f"{nome_arquivo}: existem amostras Attack != 0 no treinamento."
            )

        df["Tipo"] = "Normal"

        print(
            f"Treinamento normal: "
            f"{len(df)} amostras -> Tipo = Normal"
        )

    # ========================================================
    # TESTE
    # ========================================================

    else:

        cfg = CONFIG_TESTES[nome_arquivo]

        attack = df["Attack"].astype(int)

        inicio = (
            attack.eq(1) &
            attack.shift(fill_value=0).eq(0)
        )

        fim = (
            attack.eq(1) &
            attack.shift(-1, fill_value=0).eq(0)
        )

        idx_inicio = df.index[inicio].tolist()
        idx_fim = df.index[fim].tolist()

        if len(idx_inicio) != len(idx_fim):
            raise RuntimeError(
                f"{nome_arquivo}: números diferentes de inícios e fins."
            )

        if len(idx_inicio) != cfg["eventos"]:
            raise RuntimeError(
                f"{nome_arquivo}: encontrados {len(idx_inicio)} eventos; "
                f"esperados {cfg['eventos']}."
            )

        df["Tipo"] = "Normal"

        for numero, (ini, fim_evento) in enumerate(
            zip(idx_inicio, idx_fim),
            start=1
        ):

            evento = f"A{cfg['prefixo']}{numero:02d}"

            df.loc[ini:fim_evento, "Tipo"] = evento

            print(
                f"{evento:5s} | "
                f"{df.loc[ini, 'timestamp']} -> "
                f"{df.loc[fim_evento, 'timestamp']} | "
                f"{fim_evento - ini + 1} amostras"
            )

        ataques_attack = int((df["Attack"] == 1).sum())

        if ataques_attack != cfg["ataques"]:
            raise RuntimeError(
                f"{nome_arquivo}: Attack=1 possui {ataques_attack}; "
                f"esperado {cfg['ataques']}."
            )

        if len(df[(df["Attack"] == 1) & (df["Tipo"] == "Normal")]) != 0:
            raise RuntimeError(
                f"{nome_arquivo}: existem Attack=1 sem identificação Axxx."
            )

        if len(df[(df["Attack"] == 0) & (df["Tipo"] != "Normal")]) != 0:
            raise RuntimeError(
                f"{nome_arquivo}: existem Attack=0 identificados como Axxx."
            )

    # ========================================================
    # VALIDAÇÃO COMUM
    # ========================================================

    assert len(df) == len(original)
    assert len(df.columns) == len(colunas_originais) + 1
    assert df.columns.tolist()[:-1] == colunas_originais
    assert df.columns[-1] == "Tipo"

    pd.testing.assert_frame_equal(
        original,
        df[colunas_originais],
        check_dtype=True,
        check_exact=True,
    )

    # ========================================================
    # GRAVAÇÃO
    # ========================================================

    nome_saida = nome_arquivo.replace(".csv", "_rotulado.csv")
    arquivo_saida = SAIDA / nome_saida

    df.to_csv(arquivo_saida, index=False)

    # ========================================================
    # VALIDAÇÃO APÓS RELEITURA
    # ========================================================

    conferido = pd.read_csv(arquivo_saida)

    if len(conferido) != len(original):
        raise RuntimeError("Número de linhas mudou após salvar.")

    if conferido.columns.tolist() != df.columns.tolist():
        raise RuntimeError("Estrutura de colunas mudou após salvar.")

    if not conferido["Attack"].equals(original["Attack"]):
        raise RuntimeError("A coluna Attack mudou após salvar.")

    if not conferido["Tipo"].equals(df["Tipo"]):
        raise RuntimeError("A coluna Tipo mudou após salvar.")

    print("\nARQUIVO SALVO:")
    print(arquivo_saida)

    print("\nDISTRIBUIÇÃO DE Tipo:")
    print(
        conferido["Tipo"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nVALIDAÇÃO DO ARQUIVO SALVO: OK")


print("\nHAI 22.04 — ROTULAGEM DOS ARQUIVOS")
print("Arquivos ativos:", ARQUIVOS_ATIVOS)

for arquivo in ARQUIVOS_ATIVOS:
    rotular_arquivo(arquivo)

print("\n" + "=" * 90)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 90)
