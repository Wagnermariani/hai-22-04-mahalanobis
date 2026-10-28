from pathlib import Path

import pandas as pd


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

BASE = (
    Path(__file__).resolve().parent.parent
    / "dataset_trabalho"
    / "hai-22.04"
)

ENTRADA = (
    BASE
    / "rotulados"
)

MAPA_SEGMENTOS = (
    BASE
    / "logs"
    / "candidatas_treino_validacao"
    / "candidata_A_segmentos.csv"
)

SAIDA = (
    BASE
    / "divisao_oficial"
)

SAIDA_TREINO = (
    SAIDA
    / "treino"
)

SAIDA_VALIDACAO = (
    SAIDA
    / "validacao"
)

SAIDA_GUARDA = (
    SAIDA
    / "nao_usados"
    / "guarda"
)

SAIDA_RESIDUO = (
    SAIDA
    / "nao_usados"
    / "residuo"
)

SAIDA_LOGS = (
    SAIDA
    / "logs"
)


TRAIN_FILES = [
    "train1_rotulado.csv",
    "train2_rotulado.csv",
    "train3_rotulado.csv",
    "train4_rotulado.csv",
    "train5_rotulado.csv",
    "train6_rotulado.csv",
]


ESPERADO_GLOBAL = {
    "treino": 806046,
    "guarda": 354,
    "validacao": 198000,
    "residuo_nao_usado": 2,
}


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def preparar_diretorios():

    for diretorio in [
        SAIDA_TREINO,
        SAIDA_VALIDACAO,
        SAIDA_GUARDA,
        SAIDA_RESIDUO,
        SAIDA_LOGS,
    ]:
        diretorio.mkdir(
            parents=True,
            exist_ok=True
        )


def caminho_saida(
    nome_arquivo,
    destino
):

    base_nome = (
        Path(nome_arquivo)
        .stem
        .replace("_rotulado", "")
    )

    if destino == "treino":

        return (
            SAIDA_TREINO
            / f"{base_nome}_treino.csv"
        )

    if destino == "validacao":

        return (
            SAIDA_VALIDACAO
            / f"{base_nome}_validacao.csv"
        )

    if destino == "guarda":

        return (
            SAIDA_GUARDA
            / f"{base_nome}_guarda.csv"
        )

    if destino == "residuo_nao_usado":

        return (
            SAIDA_RESIDUO
            / f"{base_nome}_residuo.csv"
        )

    raise RuntimeError(
        f"Destino desconhecido: {destino}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 100)
    print("MATERIALIZAÇÃO DA DIVISÃO OFICIAL TREINO × VALIDAÇÃO")
    print("Candidata congelada: A")
    print("=" * 100)

    preparar_diretorios()

    if not MAPA_SEGMENTOS.exists():

        raise FileNotFoundError(
            f"Mapa não encontrado: {MAPA_SEGMENTOS}"
        )

    mapa = pd.read_csv(
        MAPA_SEGMENTOS,
        encoding="utf-8"
    )

    colunas_obrigatorias = [
        "arquivo",
        "destino",
        "linha_inicio",
        "linha_fim",
        "n_amostras",
    ]

    faltantes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in mapa.columns
    ]

    if faltantes:

        raise RuntimeError(
            f"Colunas ausentes no mapa: {faltantes}"
        )

    destinos_validos = {
        "treino",
        "guarda",
        "validacao",
        "residuo_nao_usado",
    }

    destinos_encontrados = set(
        mapa["destino"].unique()
    )

    invalidos = (
        destinos_encontrados
        - destinos_validos
    )

    if invalidos:

        raise RuntimeError(
            f"Destinos inválidos encontrados: {invalidos}"
        )

    resumo = []

    totais_globais = {
        "treino": 0,
        "guarda": 0,
        "validacao": 0,
        "residuo_nao_usado": 0,
    }

    total_original_global = 0

    esquema_referencia = None

    for nome_arquivo in TRAIN_FILES:

        print("\n" + "-" * 100)
        print(nome_arquivo)
        print("-" * 100)

        caminho_entrada = (
            ENTRADA
            / nome_arquivo
        )

        if not caminho_entrada.exists():

            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho_entrada}"
            )

        df = pd.read_csv(
            caminho_entrada,
            encoding="utf-8"
        )

        # ------------------------------------------------------------
        # Confirma consistência de esquema entre os seis arquivos.
        # ------------------------------------------------------------

        colunas = list(
            df.columns
        )

        if esquema_referencia is None:

            esquema_referencia = colunas

        elif colunas != esquema_referencia:

            raise RuntimeError(
                f"{nome_arquivo}: esquema diferente "
                "dos arquivos anteriores."
            )

        # ------------------------------------------------------------
        # Confirma que os arquivos usados para treino/validação
        # continuam sendo exclusivamente normais.
        # ------------------------------------------------------------

        if "Attack" not in df.columns:

            raise RuntimeError(
                f"{nome_arquivo}: coluna Attack ausente."
            )

        valores_attack = set(
            pd.to_numeric(
                df["Attack"],
                errors="raise"
            ).unique()
        )

        if valores_attack != {0}:

            raise RuntimeError(
                f"{nome_arquivo}: esperado apenas Attack=0; "
                f"encontrado {valores_attack}."
            )

        segmentos = (
            mapa[
                mapa["arquivo"] == nome_arquivo
            ]
            .sort_values("linha_inicio")
            .reset_index(drop=True)
        )

        if segmentos.empty:

            raise RuntimeError(
                f"Nenhum segmento encontrado para {nome_arquivo}."
            )

        # ------------------------------------------------------------
        # Validação rigorosa do mapa.
        #
        # Os segmentos devem:
        # - começar na linha zero;
        # - ser contíguos;
        # - terminar exatamente na última linha;
        # - possuir n_amostras coerente.
        # ------------------------------------------------------------

        if int(
            segmentos.iloc[0]["linha_inicio"]
        ) != 0:

            raise RuntimeError(
                f"{nome_arquivo}: primeiro segmento "
                "não começa na linha 0."
            )

        fim_anterior = -1

        soma_segmentos = 0

        for _, segmento in segmentos.iterrows():

            inicio = int(
                segmento["linha_inicio"]
            )

            fim = int(
                segmento["linha_fim"]
            )

            n_declarado = int(
                segmento["n_amostras"]
            )

            destino = str(
                segmento["destino"]
            )

            if inicio != (
                fim_anterior + 1
            ):

                raise RuntimeError(
                    f"{nome_arquivo}: quebra ou sobreposição "
                    f"antes da linha {inicio}."
                )

            n_calculado = (
                fim
                - inicio
                + 1
            )

            if n_calculado != n_declarado:

                raise RuntimeError(
                    f"{nome_arquivo}, destino {destino}: "
                    f"n_amostras declarado={n_declarado}, "
                    f"calculado={n_calculado}."
                )

            if fim >= len(df):

                raise RuntimeError(
                    f"{nome_arquivo}: segmento ultrapassa "
                    "o tamanho do arquivo."
                )

            parte = (
                df.iloc[
                    inicio:
                    fim + 1
                ]
                .copy()
            )

            if len(parte) != n_declarado:

                raise RuntimeError(
                    f"{nome_arquivo}, destino {destino}: "
                    "erro durante a extração."
                )

            arquivo_saida = caminho_saida(
                nome_arquivo,
                destino
            )

            parte.to_csv(
                arquivo_saida,
                index=False,
                encoding="utf-8"
            )

            totais_globais[
                destino
            ] += len(parte)

            soma_segmentos += len(parte)

            resumo.append({
                "arquivo_original":
                    nome_arquivo,
                "destino":
                    destino,
                "linha_inicio":
                    inicio,
                "linha_fim":
                    fim,
                "n_amostras":
                    len(parte),
                "arquivo_gerado":
                    str(arquivo_saida),
            })

            print(
                f"{destino:20s} "
                f"{len(parte):8,d} linhas  ->  "
                f"{arquivo_saida.name}"
            )

            fim_anterior = fim

        if fim_anterior != (
            len(df) - 1
        ):

            raise RuntimeError(
                f"{nome_arquivo}: último segmento termina "
                f"em {fim_anterior}, mas o arquivo termina "
                f"em {len(df) - 1}."
            )

        if soma_segmentos != len(df):

            raise RuntimeError(
                f"{nome_arquivo}: soma dos segmentos "
                f"({soma_segmentos}) != número de linhas "
                f"original ({len(df)})."
            )

        total_original_global += len(df)

        print(
            f"TOTAL ORIGINAL: {len(df):,}"
        )

        print(
            f"TOTAL MATERIALIZADO: {soma_segmentos:,}"
        )

    # =========================================================================
    # AUDITORIA GLOBAL
    # =========================================================================

    print("\n" + "=" * 100)
    print("AUDITORIA GLOBAL")
    print("=" * 100)

    total_materializado = sum(
        totais_globais.values()
    )

    for destino in [
        "treino",
        "guarda",
        "validacao",
        "residuo_nao_usado",
    ]:

        obtido = (
            totais_globais[
                destino
            ]
        )

        esperado = (
            ESPERADO_GLOBAL[
                destino
            ]
        )

        print(
            f"{destino:20s}: "
            f"{obtido:10,d} "
            f"(esperado {esperado:10,d})"
        )

        if obtido != esperado:

            raise RuntimeError(
                f"Total inesperado para {destino}: "
                f"{obtido} != {esperado}"
            )

    print("-" * 100)

    print(
        f"TOTAL MATERIALIZADO : "
        f"{total_materializado:,}"
    )

    print(
        f"TOTAL ORIGINAL       : "
        f"{total_original_global:,}"
    )

    if (
        total_materializado
        != total_original_global
    ):

        raise RuntimeError(
            "A soma dos dados materializados "
            "não coincide com a soma dos arquivos originais."
        )

    if total_original_global != 1004402:

        raise RuntimeError(
            "Número global de linhas diferente "
            "das 1.004.402 linhas previamente auditadas."
        )

    # =========================================================================
    # SALVA LOG
    # =========================================================================

    df_resumo = pd.DataFrame(
        resumo
    )

    arquivo_resumo = (
        SAIDA_LOGS
        / "resumo_materializacao_divisao_oficial.csv"
    )

    df_resumo.to_csv(
        arquivo_resumo,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print("DIVISÃO OFICIAL MATERIALIZADA COM SUCESSO")
    print("=" * 100)

    print(
        "\nTreino     : "
        f"{totais_globais['treino']:,}"
    )

    print(
        "Validação  : "
        f"{totais_globais['validacao']:,}"
    )

    print(
        "Guarda     : "
        f"{totais_globais['guarda']:,}"
    )

    print(
        "Resíduos   : "
        f"{totais_globais['residuo_nao_usado']:,}"
    )

    print(
        "Total      : "
        f"{total_original_global:,}"
    )

    print(
        f"\nLog:\n{arquivo_resumo}"
    )


if __name__ == "__main__":
    main()