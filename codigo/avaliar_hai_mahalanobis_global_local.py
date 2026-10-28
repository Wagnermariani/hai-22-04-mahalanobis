#!/usr/bin/env python3
"""Avaliacao independente HAI 22.04: Mahalanobis global x local MAX.

Protocolo predefinido: LedoitWolf float64 nos residuos normais de treino;
limiares P99 da validacao normal previamente congelados; teste somente para
avaliacao. Nao executa a RNN nem altera resultados historicos.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.covariance import LedoitWolf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / 'resultado/diagnostico_mahalanobis_normal_only'
TREINO = ROOT / 'residuos_treino_assinados.npy'
VALIDACAO = BASE / 'resultado/rnn_hai/secaoMONOLITICO/residuos_monolitico/residuos_validacao_monolitico.csv'
TESTE = BASE / 'resultado/rnn_hai/secaoMONOLITICO/residuos_monolitico/residuos_teste_monolitico.csv'
CABECALHO = BASE / 'dataset_trabalho/hai-22.04/normalizados_oficiais/treino/train1_treino_normalizado.csv'
DETALHE = ROOT / 'diagnostico/diagnostico_por_grupo.csv'
RESUMO = ROOT / 'diagnostico/diagnostico_resumo.csv'
DESTINO = ROOT / 'avaliacao_ataques'
HASHES = {
    DETALHE: '5d9713a1d89065585e66dc02dfb4a42646be646779b615c55430a29951430901',
    RESUMO: 'ac87726c478f6273eb4d5971828b41270cbfda0b61637d767801da5d04bdaf1a',
}
TAMANHOS = {'P1': 44, 'P2': 24, 'P3': 7, 'P4': 11}
FPR_ALVO = 0.01
N_VAL = 197646
N_TREINO = 805692
N_ATAQUES = 12030
N_EVENTOS = 58
BLOCO_TESTE = 10000


def exigir(condicao, msg):
    if not condicao:
        raise RuntimeError(msg)


def sha256(arquivo):
    h = hashlib.sha256()
    with arquivo.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def obter_features():
    colunas = pd.read_csv(CABECALHO, nrows=0).columns.str.strip().tolist()
    features = [c for c in colunas if c not in ('timestamp', 'Attack', 'Tipo')]
    exigir(len(features) == 86 and len(set(features)) == 86,
           'Lista de 86 features invalida.')
    indices = {}
    for grupo, tamanho in TAMANHOS.items():
        idx = np.array([i for i, f in enumerate(features) if f.startswith(f'{grupo}_')], dtype=np.int64)
        exigir(len(idx) == tamanho, f'Grupo {grupo} invalido: {len(idx)} features.')
        indices[grupo] = idx
    todos = np.concatenate(list(indices.values()))
    exigir(np.array_equal(np.sort(todos), np.arange(86)), 'Particao incompleta/sobreposta.')
    return features, indices


def metricas(contagem):
    tn, fp, fn, tp = [int(v) for v in contagem.ravel()]
    # Mesmas definicoes do baseline historico, calculadas de forma equivalente
    # a partir da matriz, evitando armazenar todas as previsoes do teste.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'F1': f1,
            'MCC': mcc, 'FPR': fpr}


def main():
    # Falha ANTES de qualquer trabalho caso o destino ja exista.
    exigir(not DESTINO.exists() and not DESTINO.is_symlink(),
           f'Destino ja existe, nada sera sobrescrito: {DESTINO}')

    print('=== CONFERENCIA DO DIAGNOSTICO CONGELADO ===', flush=True)
    for arquivo, esperado in HASHES.items():
        obtido = sha256(arquivo)
        exigir(obtido == esperado, f'Hash divergente: {arquivo}\n{obtido}')
        print(f'SHA-256 OK: {arquivo.name}', flush=True)

    resumo = pd.read_csv(RESUMO)
    exigir(len(resumo) == 1, 'Resumo deve conter uma linha.')
    r = resumo.iloc[0]
    exigir(r['Dataset'] == 'HAI_22.04' and r['Espaco'] == 'Residuos_RNN'
           and r['Precisao_Estatistica'] == 'float64' and int(r['N_Grupos']) == 4
           and np.isclose(float(r['FPR_Alvo']), FPR_ALVO, rtol=0, atol=1e-15),
           'Resumo nao corresponde ao protocolo predefinido.')
    tau_global = float(r['Tau_Global'])
    tau_local = float(r['Tau_Local'])
    alarmes_g = int(r['Alarmes_Val_Global'])
    alarmes_l = int(r['Alarmes_Val_Local'])
    exigir(alarmes_g == 1977 and alarmes_l == 1977, 'Orcamento de validacao inesperado.')
    exigir(np.isclose(tau_global, 199.4393418878611, rtol=0, atol=1e-10)
           and np.isclose(tau_local, 135.92393580193848, rtol=0, atol=1e-10),
           'Limiares congelados inesperados.')

    features, indices = obter_features()
    colunas_err = [f'err_{c}' for c in features]

    print('Carregando treino normal e validacao normal...', flush=True)
    X_train = np.load(TREINO, allow_pickle=False)
    exigir(X_train.shape == (N_TREINO, 86) and X_train.dtype == np.float32
           and np.isfinite(X_train).all(), 'Residuos normais de treino invalidos.')

    val = pd.read_csv(VALIDACAO, usecols=colunas_err + ['Attack'],
                      dtype={c: np.float32 for c in colunas_err})
    exigir(list(val[colunas_err].columns) == colunas_err, 'Ordem da validacao divergente.')
    exigir(len(val) == N_VAL, 'Quantidade de amostras de validacao inesperada.')
    exigir(pd.to_numeric(val['Attack'], errors='raise').eq(0).all(),
           'Validacao nao e exclusivamente normal.')
    X_val = val[colunas_err].to_numpy(dtype=np.float64)
    exigir(np.isfinite(X_val).all(), 'Validacao contem residuo nao finito.')
    del val
    X_train = X_train.astype(np.float64)

    print('Ajustando LedoitWolf global e quatro locais em float64...', flush=True)
    modelo_global = LedoitWolf(store_precision=True, assume_centered=False)
    modelo_global.fit(X_train)
    modelos = {}
    for grupo, idx in indices.items():
        m = LedoitWolf(store_precision=True, assume_centered=False)
        m.fit(X_train[:, idx])
        modelos[grupo] = m
        print(f'  {grupo}: {len(idx)} features', flush=True)
    del X_train

    # Reproducao exata dos quantis e contagens do diagnostico ANTES do teste.
    sg_val = modelo_global.mahalanobis(X_val)
    sl_val = np.max(np.column_stack([
        modelos[g].mahalanobis(X_val[:, idx]) for g, idx in indices.items()
    ]), axis=1)
    tau_g_refeito = float(np.quantile(sg_val, 1.0 - FPR_ALVO))
    tau_l_refeito = float(np.quantile(sl_val, 1.0 - FPR_ALVO))
    n_g = int(np.count_nonzero(sg_val > tau_global))
    n_l = int(np.count_nonzero(sl_val > tau_local))
    exigir(np.isclose(tau_g_refeito, tau_global, rtol=0, atol=1e-8)
           and np.isclose(tau_l_refeito, tau_local, rtol=0, atol=1e-8)
           and n_g == alarmes_g and n_l == alarmes_l,
           'Falha em reproduzir os limiares/contagens de validacao congelados.')
    print(f'Validacao reproduzida: global={n_g}/{N_VAL}, local={n_l}/{N_VAL}', flush=True)
    print(f'Limiar global: {tau_global:.15f}; local MAX: {tau_local:.15f}', flush=True)
    del X_val, sg_val, sl_val

    print('=== AVALIANDO O TESTE, SEM RECALIBRACAO ===', flush=True)
    colunas_meta = ['arquivo_origem', 'Tipo', 'Attack']
    cont_g = np.zeros((2, 2), dtype=np.int64)
    cont_l = np.zeros((2, 2), dtype=np.int64)
    eventos = {}  # Tipo -> contagens e deteccoes
    total = 0
    n_ataques = 0
    intervalos = 0
    anterior = None  # (arquivo_origem, attack, Tipo)

    leitor = pd.read_csv(TESTE, usecols=colunas_meta + colunas_err,
                         dtype={**{c: np.float32 for c in colunas_err},
                                'Tipo': 'string', 'arquivo_origem': 'string'},
                         chunksize=BLOCO_TESTE)
    for bloco_num, df in enumerate(leitor, 1):
        exigir(list(df[colunas_err].columns) == colunas_err, 'Ordem dos residuos de teste divergente.')
        X = df[colunas_err].to_numpy(dtype=np.float64)
        exigir(np.isfinite(X).all(), 'Teste contem residuos nao finitos.')
        ataque_numerico = pd.to_numeric(df['Attack'], errors='raise')
        exigir(ataque_numerico.notna().all(), 'Rotulo Attack ausente no teste.')
        y = ataque_numerico.ne(0).to_numpy(dtype=bool)
        tipo = df['Tipo'].astype('string')
        origem = df['arquivo_origem'].astype('string')
        exigir((~tipo[y].isna()).all() and tipo[y].str.strip().ne('').all(),
               'Amostra de ataque sem Tipo.')
        exigir(origem.notna().all(), 'Origem ausente no teste.')

        sg = modelo_global.mahalanobis(X)
        sl = np.max(np.column_stack([
            modelos[g].mahalanobis(X[:, idx]) for g, idx in indices.items()
        ]), axis=1)
        pg = sg > tau_global
        pl = sl > tau_local

        cont_g[0, 0] += np.count_nonzero(~y & ~pg)
        cont_g[0, 1] += np.count_nonzero(~y & pg)
        cont_g[1, 0] += np.count_nonzero(y & ~pg)
        cont_g[1, 1] += np.count_nonzero(y & pg)
        cont_l[0, 0] += np.count_nonzero(~y & ~pl)
        cont_l[0, 1] += np.count_nonzero(~y & pl)
        cont_l[1, 0] += np.count_nonzero(y & ~pl)
        cont_l[1, 1] += np.count_nonzero(y & pl)

        # Contar intervalos considerando transicoes ENTRE blocos.
        for i in range(len(df)):
            estado = (str(origem.iloc[i]), bool(y[i]), str(tipo.iloc[i]))
            if estado[1]:
                chave = estado[2]
                info = eventos.setdefault(chave, {'Amostras_Ataque': 0,
                    'Detectado_Global': False, 'Detectado_Local': False,
                    'Alarmes_Global': 0, 'Alarmes_Local': 0})
                info['Amostras_Ataque'] += 1
                info['Alarmes_Global'] += int(pg[i])
                info['Alarmes_Local'] += int(pl[i])
                info['Detectado_Global'] |= bool(pg[i])
                info['Detectado_Local'] |= bool(pl[i])
                if anterior != estado:
                    intervalos += 1
            anterior = estado

        total += len(df)
        n_ataques += int(np.count_nonzero(y))
        if bloco_num % 10 == 0:
            print(f'  {total} linhas do teste avaliadas...', flush=True)

    exigir(n_ataques == N_ATAQUES, f'Amostras de ataque inesperadas: {n_ataques}')
    exigir(len(eventos) == N_EVENTOS and intervalos == N_EVENTOS,
           f'Eventos inesperados: IDs={len(eventos)}, intervalos={intervalos}')
    exigir(cont_g.sum() == total and cont_l.sum() == total, 'Matrizes nao fecham.')
    exigir(cont_g[1].sum() == n_ataques and cont_l[1].sum() == n_ataques,
           'Contagem de ataque divergente.')

    mg = metricas(cont_g)
    ml = metricas(cont_l)
    linhas_eventos = [{'Evento': k, **v} for k, v in sorted(eventos.items())]
    df_eventos = pd.DataFrame(linhas_eventos)
    rg = int(df_eventos['Detectado_Global'].sum())
    rl = int(df_eventos['Detectado_Local'].sum())

    df_resumo = pd.DataFrame([
        {'Detector': 'Mahalanobis_Global', **mg, 'Eventos_Detectados': rg,
         'Total_Eventos': N_EVENTOS, 'Taxa_Deteccao_Eventos': rg/N_EVENTOS,
         'Tau_Congelado': tau_global, 'Alarmes_Val': alarmes_g,
         'N_Val': N_VAL, 'N_Teste': total, 'FPR_Alvo_Validacao': FPR_ALVO},
        {'Detector': 'Mahalanobis_Local_Max', **ml, 'Eventos_Detectados': rl,
         'Total_Eventos': N_EVENTOS, 'Taxa_Deteccao_Eventos': rl/N_EVENTOS,
         'Tau_Congelado': tau_local, 'Alarmes_Val': alarmes_l,
         'N_Val': N_VAL, 'N_Teste': total, 'FPR_Alvo_Validacao': FPR_ALVO},
    ])
    registro = {
        'dataset': 'HAI_22.04', 'origem_escores': 'Residuos assinados da RNN monolitica',
        'precisao_ajuste': 'float64', 'modelo': 'LedoitWolf(store_precision=True, assume_centered=False)',
        'grupos': TAMANHOS, 'fusao_local': 'MAX',
        'calibracao': 'P99 da validacao exclusivamente normal',
        'comparador': 'score > tau', 'limiares': {'Global': tau_global, 'Local_MAX': tau_local},
        'limiares_reproduzidos': {'Global': tau_g_refeito, 'Local_MAX': tau_l_refeito},
        'amostras_treino_normal': N_TREINO, 'amostras_validacao_normal': N_VAL,
        'amostras_teste': total, 'amostras_ataque': n_ataques,
        'eventos': N_EVENTOS,
        'versoes': {'python': sys.version.split()[0], 'numpy': np.__version__,
                   'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'sha256_diagnostico': {p.name: h for p, h in HASHES.items()},
        'observacao': 'FPR-alvo de 1% refere-se a validacao; FPR de teste e resultado, nao orcamento recalibrado.',
    }

    # Escrita nova e exclusiva SOMENTE apos sucesso de todos os calculos.
    DESTINO.mkdir(parents=False, exist_ok=False)
    with (DESTINO / 'metricas_global_local.csv').open('x', encoding='utf-8') as f:
        df_resumo.to_csv(f, index=False)
    with (DESTINO / 'comparacao_58_eventos.csv').open('x', encoding='utf-8') as f:
        df_eventos.to_csv(f, index=False)
    with (DESTINO / 'protocolo.json').open('x', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

    print('\n=== RESULTADOS ===')
    print(df_resumo.to_string(index=False))
    print(f'\nEventos global: {rg}/{N_EVENTOS}; local: {rl}/{N_EVENTOS}')
    print(f'Arquivos novos, exclusivamente em: {DESTINO}')


if __name__ == '__main__':
    main()
