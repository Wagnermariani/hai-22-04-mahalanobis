import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE = (
    Path(__file__).resolve().parent.parent
    / "dataset_trabalho"
    / "hai-22.04"
)

ENTRADA_TREINO = (
    BASE
    / "divisao_oficial"
    / "treino"
)

ENTRADA_VALIDACAO = (
    BASE
    / "divisao_oficial"
    / "validacao"
)

ENTRADA_TESTE = (
    BASE
    / "rotulados"
)

SAIDA = (
    BASE
    / "normalizados_oficiais"
)

SAIDA_TREINO = (
    SAIDA
    / "treino"
)

SAIDA_VALIDACAO = (
    SAIDA
    / "validacao"
)

SAIDA_TESTE = (
    SAIDA
    / "teste"
)

LOGS = (
    SAIDA
    / "logs"
)

for diretorio in [
    SAIDA_TREINO,
    SAIDA_VALIDACAO,
    SAIDA_TESTE,
    LOGS,
]:
    diretorio.mkdir(
        parents=True,
        exist_ok=True
    )



SAIDA.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# ============================================================
# BOOLEANOS DOCUMENTADOS
# ============================================================

BOOLEANOS = [
    "P1_PP01AD",
    "P1_PP01AR",
    "P1_PP01BD",
    "P1_PP01BR",
    "P1_PP02D",
    "P1_PP02R",
    "P1_SOL01D",
    "P1_SOL03D",
    "P1_STSP",
    "P2_ATSW_Lamp",
    "P2_AutoGO",
    "P2_Emerg",
    "P2_MASW",
    "P2_MASW_Lamp",
    "P2_ManualGO",
    "P2_OnOff",
    "P2_TripEx",
]

TREINOS = [
    "train1_treino.csv",
    "train2_treino.csv",
    "train3_treino.csv",
    "train4_treino.csv",
    "train5_treino.csv",
    "train6_treino.csv",
]

VALIDACOES = [
    "train1_validacao.csv",
    "train2_validacao.csv",
    "train3_validacao.csv",
    "train4_validacao.csv",
    "train5_validacao.csv",
    "train6_validacao.csv",
]

TESTES = [
    "test1_rotulado.csv",
    "test2_rotulado.csv",
    "test3_rotulado.csv",
    "test4_rotulado.csv",
]


# ============================================================
# DESCOBRIR E VALIDAR AS 86 FEATURES
# ============================================================

referencia = pd.read_csv(
    ENTRADA_TREINO / TREINOS[0],
    nrows=5
)

EXCLUIR = ["timestamp", "Attack", "Tipo"]

FEATURES = [
    c for c in referencia.columns
    if c not in EXCLUIR
]

if len(FEATURES) != 86:
    raise RuntimeError(
        f"Encontradas {len(FEATURES)} features; esperado = 86."
    )

faltando = sorted(set(BOOLEANOS) - set(FEATURES))

if faltando:
    raise RuntimeError(
        f"Booleanos não encontrados: {faltando}"
    )

NUMERICAS = [
    c for c in FEATURES
    if c not in BOOLEANOS
]

if len(BOOLEANOS) != 17:
    raise RuntimeError("Quantidade de Booleanos diferente de 17.")

if len(NUMERICAS) != 69:
    raise RuntimeError(
        f"Encontradas {len(NUMERICAS)} numéricas; esperado = 69."
    )

print("\nHAI 22.04 — NORMALIZAÇÃO")
print("Features:", len(FEATURES))
print("Booleanos preservados:", len(BOOLEANOS))
print("Numéricas MinMax:", len(NUMERICAS))

# ============================================================
# AJUSTE DO MINMAX SOMENTE NO TREINO OFICIAL
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, 1),
    clip=False
)

print(
    "\nAJUSTANDO MINMAX SOMENTE "
    "NO TREINO OFICIAL:"
)

total_treino_ajuste = 0

for nome in TREINOS:

    caminho = (
        ENTRADA_TREINO
        / nome
    )

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado: {caminho}"
        )

    df = pd.read_csv(
        caminho,
        usecols=NUMERICAS
    )

    scaler.partial_fit(
        df[NUMERICAS]
    )

    total_treino_ajuste += len(df)

    print(
        f"  {nome}: "
        f"{len(df):,} amostras"
    )

if total_treino_ajuste != 806046:
    raise RuntimeError(
        "Quantidade de amostras usada para ajustar "
        f"o MinMax = {total_treino_ajuste:,}; "
        "esperado = 806.046."
    )

print(
    "\nTotal usado no ajuste do MinMax: "
    f"{total_treino_ajuste:,}"
)

print(
    "MinMax ajustado exclusivamente "
    "no treino oficial."
)

# ============================================================
# SALVAR PARÂMETROS APRENDIDOS
# ============================================================

parametros = pd.DataFrame({
    "variavel": NUMERICAS,
    "train_min": scaler.data_min_,
    "train_max": scaler.data_max_,
    "train_range": scaler.data_range_,
    "constante_no_treino": scaler.data_range_ == 0,
})

arquivo_parametros = LOGS / "parametros_minmax_hai_22_04.csv"
parametros.to_csv(arquivo_parametros, index=False)

print("\nParâmetros salvos em:")
print(arquivo_parametros)

print("\nVariáveis numéricas constantes no treino:")
print(
    parametros.loc[
        parametros["constante_no_treino"],
        ["variavel", "train_min", "train_max"]
    ].to_string(index=False)
)

# ============================================================
# TRANSFORMAÇÃO DOS CONJUNTOS OFICIAIS
# ============================================================

CONJUNTOS = [
    {
        "nome": "treino",
        "entrada": ENTRADA_TREINO,
        "saida": SAIDA_TREINO,
        "arquivos": TREINOS,
        "total_esperado": 806046,
    },
    {
        "nome": "validacao",
        "entrada": ENTRADA_VALIDACAO,
        "saida": SAIDA_VALIDACAO,
        "arquivos": VALIDACOES,
        "total_esperado": 198000,
    },
    {
        "nome": "teste",
        "entrada": ENTRADA_TESTE,
        "saida": SAIDA_TESTE,
        "arquivos": TESTES,
        "total_esperado": 361200,
    },
]


resumo_normalizacao = []


for conjunto in CONJUNTOS:

    nome_conjunto = conjunto["nome"]
    diretorio_entrada = conjunto["entrada"]
    diretorio_saida = conjunto["saida"]
    arquivos = conjunto["arquivos"]
    total_esperado = conjunto["total_esperado"]

    print("\n" + "#" * 90)
    print(
        f"CONJUNTO: {nome_conjunto.upper()}"
    )
    print("#" * 90)

    total_conjunto = 0

    for nome in arquivos:

        print("\n" + "=" * 90)
        print(
            f"TRANSFORMANDO: {nome}"
        )
        print("=" * 90)

        entrada = (
            diretorio_entrada
            / nome
        )

        if not entrada.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {entrada}"
            )

        df = pd.read_csv(
            entrada
        )

        original = df.copy(
            deep=True
        )

        # --------------------------------------------------------
        # Estrutura obrigatória
        # --------------------------------------------------------

        if (
            df.columns.tolist()
            != referencia.columns.tolist()
        ):
            raise RuntimeError(
                f"{nome}: esquema de colunas "
                "diferente do esperado."
            )

        if len(df) == 0:
            raise RuntimeError(
                f"{nome}: arquivo vazio."
            )

        # --------------------------------------------------------
        # Confirma integridade das numéricas antes do MinMax
        # --------------------------------------------------------

        valores_antes = (
            df[NUMERICAS]
            .to_numpy()
        )

        if np.isnan(
            valores_antes
        ).any():
            raise RuntimeError(
                f"{nome}: NaN presente "
                "antes da normalização."
            )

        if np.isinf(
            valores_antes
        ).any():
            raise RuntimeError(
                f"{nome}: Inf presente "
                "antes da normalização."
            )

        # --------------------------------------------------------
        # Preserva booleanos para comparação posterior
        # --------------------------------------------------------

        booleanos_antes = (
            df[BOOLEANOS]
            .copy()
        )

        # --------------------------------------------------------
        # MinMax:
        #
        # O scaler já foi ajustado EXCLUSIVAMENTE
        # nas 806.046 linhas do treino oficial.
        #
        # O mesmo scaler é aplicado, sem novo fit, a:
        # - treino;
        # - validação;
        # - teste.
        # --------------------------------------------------------

        df[NUMERICAS] = (
            scaler.transform(
                df[NUMERICAS]
            )
        )

        # --------------------------------------------------------
        # VALIDAÇÕES APÓS A TRANSFORMAÇÃO
        # --------------------------------------------------------

        # Timestamp / Attack / Tipo devem permanecer iguais
        for col in [
            "timestamp",
            "Attack",
            "Tipo",
        ]:

            if not df[col].equals(
                original[col]
            ):
                raise RuntimeError(
                    f"{nome}: coluna {col} "
                    "foi alterada."
                )

        # Booleanos devem permanecer exatamente iguais
        pd.testing.assert_frame_equal(
            booleanos_antes,
            df[BOOLEANOS],
            check_dtype=True,
            check_exact=True,
        )

        # Nenhuma feature removida
        if len(FEATURES) != 86:
            raise RuntimeError(
                "Número de features mudou."
            )

        # Nenhum NaN/Inf criado nas numéricas
        valores_depois = (
            df[NUMERICAS]
            .to_numpy()
        )

        if np.isnan(
            valores_depois
        ).any():
            raise RuntimeError(
                f"{nome}: NaN criado "
                "pela normalização."
            )

        if np.isinf(
            valores_depois
        ).any():
            raise RuntimeError(
                f"{nome}: Inf criado "
                "pela normalização."
            )

        # --------------------------------------------------------
        # NOME DO ARQUIVO DE SAÍDA
        # --------------------------------------------------------

        if nome.endswith(
            "_treino.csv"
        ):

            nome_saida = nome.replace(
                "_treino.csv",
                "_treino_normalizado.csv"
            )

        elif nome.endswith(
            "_validacao.csv"
        ):

            nome_saida = nome.replace(
                "_validacao.csv",
                "_validacao_normalizado.csv"
            )

        elif nome.endswith(
            "_rotulado.csv"
        ):

            nome_saida = nome.replace(
                "_rotulado.csv",
                "_normalizado.csv"
            )

        else:

            raise RuntimeError(
                f"Nome de arquivo inesperado: {nome}"
            )

        arquivo_saida = (
            diretorio_saida
            / nome_saida
        )

        # --------------------------------------------------------
        # GRAVAÇÃO
        # --------------------------------------------------------

        df.to_csv(
            arquivo_saida,
            index=False
        )

        # --------------------------------------------------------
        # RESUMO
        # --------------------------------------------------------

        minimo = (
            df[NUMERICAS]
            .min()
            .min()
        )

        maximo = (
            df[NUMERICAS]
            .max()
            .max()
        )

        abaixo_zero = int(
            (
                df[NUMERICAS] < 0
            )
            .sum()
            .sum()
        )

        acima_um = int(
            (
                df[NUMERICAS] > 1
            )
            .sum()
            .sum()
        )

        total_conjunto += len(df)

        resumo_normalizacao.append({
            "conjunto":
                nome_conjunto,
            "arquivo_entrada":
                str(entrada),
            "arquivo_saida":
                str(arquivo_saida),
            "linhas":
                len(df),
            "colunas":
                len(df.columns),
            "minimo_normalizado":
                minimo,
            "maximo_normalizado":
                maximo,
            "valores_abaixo_zero":
                abaixo_zero,
            "valores_acima_um":
                acima_um,
        })

        print(
            "Arquivo salvo:"
        )

        print(
            arquivo_saida
        )

        print(
            f"Linhas: {len(df):,}"
        )

        print(
            f"Colunas: {len(df.columns)}"
        )

        print(
            "Menor valor numérico normalizado: "
            f"{minimo}"
        )

        print(
            "Maior valor numérico normalizado: "
            f"{maximo}"
        )

        print(
            f"Valores < 0: {abaixo_zero}"
        )

        print(
            f"Valores > 1: {acima_um}"
        )

        print(
            "Timestamp preservado: OK"
        )

        print(
            "Attack preservado: OK"
        )

        print(
            "Tipo preservado: OK"
        )

        print(
            "17 Booleanos preservados: OK"
        )

        print(
            "86 features preservadas: OK"
        )

    # ------------------------------------------------------------
    # Total do conjunto
    # ------------------------------------------------------------

    print("\n" + "-" * 90)

    print(
        f"TOTAL {nome_conjunto.upper()}: "
        f"{total_conjunto:,}"
    )

    print(
        f"ESPERADO: "
        f"{total_esperado:,}"
    )

    if (
        total_conjunto
        != total_esperado
    ):
        raise RuntimeError(
            f"Total do conjunto {nome_conjunto} "
            f"= {total_conjunto:,}; "
            f"esperado = {total_esperado:,}."
        )


# ============================================================
# SALVAR RESUMO DA NORMALIZAÇÃO
# ============================================================

df_resumo_normalizacao = pd.DataFrame(
    resumo_normalizacao
)

arquivo_resumo_normalizacao = (
    LOGS
    / "resumo_normalizacao_oficial.csv"
)

df_resumo_normalizacao.to_csv(
    arquivo_resumo_normalizacao,
    index=False
)


# ============================================================
# RESUMO FINAL
# ============================================================

print("\n" + "=" * 90)
print("NORMALIZAÇÃO OFICIAL DO HAI 22.04 CONCLUÍDA")
print("=" * 90)

print(
    "\nScaler ajustado em: "
    "806.046 linhas de treino"
)

print(
    "Validação transformada: "
    "198.000 linhas"
)

print(
    "Teste transformado: "
    "361.200 linhas"
)

print(
    "\nNenhuma amostra de validação ou teste "
    "foi usada no ajuste do MinMax."
)

print(
    "\nResumo salvo em:"
)

print(
    arquivo_resumo_normalizacao
)

print(
    "\nParâmetros MinMax salvos em:"
)

print(
    arquivo_parametros
)
