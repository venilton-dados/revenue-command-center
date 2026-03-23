from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import os
import re
from dotenv import load_dotenv
from google import genai

# ==============================
# CONFIGURAÇÕES INICIAIS
# ==============================
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ==============================
# CARREGAMENTO DE DADOS
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "vendas_internet_5anos.xlsx"

df = pd.read_excel(DATA_PATH)
df["Data"] = pd.to_datetime(
    df["Data"], format="mixed", dayfirst=True, errors="coerce"
)
df = df.dropna(subset=["Data"])

# ==============================
# UTILITÁRIOS
# ==============================
def apply_filters(data, canal=None, status=None, vendedor=None):
    df_filtered = data.copy()
    if canal:
        df_filtered = df_filtered[df_filtered["Canal"] == canal]
    if status:
        df_filtered = df_filtered[df_filtered["Status"] == status]
    if vendedor:
        df_filtered = df_filtered[df_filtered["Vendedor"] == vendedor]
    return df_filtered


def get_kpis_from_df(df_filtered):
    receita_total = df_filtered["Valor (R$)"].sum()
    total_vendas = len(df_filtered)
    ticket_medio = receita_total / total_vendas if total_vendas > 0 else 0

    df_sorted = df_filtered.sort_values("Data")
    metade = len(df_sorted) // 2
    periodo_antigo = df_sorted.iloc[:metade]
    periodo_recente = df_sorted.iloc[metade:]

    receita_antiga = periodo_antigo["Valor (R$)"].sum()
    receita_recente = periodo_recente["Valor (R$)"].sum()

    crescimento = 0
    if receita_antiga > 0:
        crescimento = ((receita_recente - receita_antiga) / receita_antiga) * 100

    return {
        "receita_total": float(receita_total),
        "total_vendas": int(total_vendas),
        "ticket_medio": float(ticket_medio),
        "crescimento": float(crescimento),
    }


def to_br_currency(value):
    return f"R$ {value:,.2f}"


def find_last_sentence_end(text):
    """
    Encontra a posição do último ponto final REAL de frase,
    ignorando pontos decimais (ex: 22.76 ou R$ 1.500,00).
    Um ponto é final de frase quando:
      - NÃO está entre dois dígitos (não é decimal)
      - NÃO é seguido imediatamente por um dígito
    """
    best = -1
    for match in re.finditer(r'[.!?]', text):
        pos = match.start()
        char = text[pos]
        before = text[pos - 1] if pos > 0 else ""
        after = text[pos + 1] if pos + 1 < len(text) else ""

        # Ignora ponto decimal: dígito ANTES e dígito DEPOIS (ex: 22.76)
        if char == "." and before.isdigit() and after.isdigit():
            continue

        # Ignora ponto de abreviação numérica: dígito antes e espaço/fim depois
        # mas apenas se NÃO vier logo após uma palavra (ex: "crescimento.")
        # Regra: só considera ponto final se antes houver letra ou % ou )
        if char == "." and before.isdigit() and (after == "" or after == " "):
            # Ex: "22.76%" ou final de número — não é fim de frase elegante
            # Só aceita se realmente não tem mais nada útil
            continue

        best = pos

    return best


def safe_sentence(text, max_len=180):
    clean = re.sub(r"\s+", " ", str(text or "")).strip()

    if not clean:
        return ""

    # NÃO corta valores monetários nem números
    if len(clean) <= max_len:
        if clean[-1] not in ".!?":
            clean = clean.rstrip(" ,;:") + "."
        return clean

    # tenta encontrar último ponto real de frase
    last_dot = clean.rfind(". ", 0, max_len)

    if last_dot > int(max_len * 0.4):
        return clean[: last_dot + 1].strip()

    # fallback: NÃO cortar números
    words = clean[:max_len].split(" ")

    safe_words = []
    for w in words:
        # evita cortar números tipo 739.505,94
        if re.match(r"\d+[.,]?\d*", w):
            safe_words.append(w)
        else:
            safe_words.append(w)

    result = " ".join(safe_words)

    # remove finais ruins
    result = re.sub(
        r"\s+(de|do|da|dos|das|e|com|para|em|o|a|os|as)$",
        "",
        result,
        flags=re.IGNORECASE,
    )

    return result.strip() + "."


def split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip(" -•*\n\t") for p in parts if p.strip(" -•*\n\t")]


def extract_section(text, labels):
    if not text:
        return ""
    escaped = [re.escape(label) for label in labels]
    pattern = rf"(?is)(?:{'|'.join(escaped)})\s*(.*?)(?=\n\s*(?:\[?[A-ZÇÕÃ_ ]+\]?[:])|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def extract_bullets(section_text):
    if not section_text:
        return []
    lines = []
    for raw in section_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-•*]\s*", "", line).strip()
        if line:
            lines.append(line)
    if lines:
        return lines
    return split_sentences(section_text)


def build_fallback_sections(kpis, top_vendedores_dict, revenue_by_channel_dict, status_dict):
    top_channel = next(iter(revenue_by_channel_dict.items()), None)
    top_vendedor = next(iter(top_vendedores_dict.items()), None)
    top_status = next(iter(status_dict.items()), None)

    crescimento_fmt = f"{kpis['crescimento']:.1f}%"

    # Resumo executivo baseado em dados reais — frases completas
    if kpis["crescimento"] > 15:
        resumo = (
            f"A operação registra crescimento expressivo de {crescimento_fmt} "
            f"com receita de {to_br_currency(kpis['receita_total'])}, "
            f"indicando forte tração comercial e espaço para ampliar escala."
        )
    elif kpis["crescimento"] > 0:
        resumo = (
            f"Com receita de {to_br_currency(kpis['receita_total'])} e crescimento de {crescimento_fmt}, "
            f"a operação avança de forma consistente com espaço para acelerar nos canais líderes."
        )
    elif kpis["crescimento"] == 0:
        resumo = (
            f"Receita de {to_br_currency(kpis['receita_total'])} estável com {kpis['total_vendas']} vendas, "
            f"sinalizando necessidade de ações para retomar crescimento."
        )
    else:
        resumo = (
            f"A operação registra queda de {abs(kpis['crescimento']):.1f}% na receita, "
            f"exigindo revisão imediata dos canais e estratégia comercial."
        )

    # Pontos com dados reais
    pontos = [
        f"Receita total de {to_br_currency(kpis['receita_total'])} com ticket médio de {to_br_currency(kpis['ticket_medio'])} sustentam o volume atual.",
        f"A distribuição entre canais e vendedores indica onde concentrar a próxima alavanca comercial.",
        f"O crescimento de {crescimento_fmt} sinaliza {'oportunidade de aceleração' if kpis['crescimento'] >= 0 else 'necessidade de correção de rota'}.",
    ]

    if top_channel:
        canal_nome, canal_valor = top_channel
        pontos[1] = f"{canal_nome} lidera com {to_br_currency(canal_valor)} e deve ser o foco principal de alocação de esforço."

    if top_vendedor:
        vendedor_nome, vendedor_valor = top_vendedor
        pontos[0] = f"{vendedor_nome} lidera o ranking com {to_br_currency(vendedor_valor)}, referência para replicar no restante do time."

    if top_status:
        status_nome, status_qtd = top_status
        pontos[2] = f"O status {status_nome} concentra {int(status_qtd)} vendas e é o principal driver de volume do período."

    # Ações diretas
    acoes = [
        f"Priorizar {top_channel[0] if top_channel else 'o canal líder'} com budget incremental para capturar ganho imediato.",
        f"Replicar o método de {top_vendedor[0] if top_vendedor else 'o vendedor líder'} nos demais para elevar a média do time.",
        "Testar upsell e cross-sell nos clientes ativos para aumentar ticket médio sem custo de aquisição.",
    ]

    return {
        "resumo": safe_sentence(resumo, 220),
        "pontos": [safe_sentence(p, 150) for p in pontos[:3]],
        "acoes": [safe_sentence(a, 150) for a in acoes[:3]],
    }


def normalize_ai_answer(answer, fallback_sections):
    text = (answer or "").replace("\r", "").strip()

    if not text:
        return (
            "[RESUMO]\n"
            f"{fallback_sections['resumo']}\n\n"
            "[PONTOS]\n"
            + "\n".join(f"- {item}" for item in fallback_sections["pontos"])
            + "\n\n[ACOES]\n"
            + "\n".join(f"- {item}" for item in fallback_sections["acoes"])
        )

    resumo_text = extract_section(text, ["[RESUMO]", "RESUMO:", "Resumo executivo:", "Resumo:"])
    pontos_text = extract_section(text, ["[PONTOS]", "PONTOS:", "Pontos principais:", "Pontos:"])
    acoes_text = extract_section(text, ["[ACOES]", "AÇÕES RECOMENDADAS:", "Ações recomendadas:", "ACOES:", "Ações:"])

    resumo = safe_sentence(resumo_text, 220) if resumo_text else ""
    pontos = [safe_sentence(x, 150) for x in extract_bullets(pontos_text)[:3]]
    acoes = [safe_sentence(x, 150) for x in extract_bullets(acoes_text)[:3]]

    if not resumo:
        sentences = split_sentences(text)
        if sentences:
            resumo = safe_sentence(sentences[0], 220)

    if not pontos:
        sentences = split_sentences(text)
        if len(sentences) > 1:
            pontos = [safe_sentence(x, 150) for x in sentences[1:4]]

    if not acoes:
        action_candidates = []
        for sentence in split_sentences(text):
            if re.search(
                r"(prioriz|replic|test|ajust|monitor|reduz|aument|expand|focar|direcion|revis)",
                sentence,
                flags=re.IGNORECASE,
            ):
                action_candidates.append(sentence)
        acoes = [safe_sentence(x, 150) for x in action_candidates[:3]]

    if not resumo:
        resumo = fallback_sections["resumo"]
    if len(pontos) == 0:
        pontos = fallback_sections["pontos"]
    if len(acoes) == 0:
        acoes = fallback_sections["acoes"]

    pontos = pontos[:3]
    acoes = acoes[:3]

    while len(pontos) < 3:
        pontos.append(fallback_sections["pontos"][len(pontos)])
    while len(acoes) < 3:
        acoes.append(fallback_sections["acoes"][len(acoes)])

    return (
        "[RESUMO]\n"
        f"{resumo}\n\n"
        "[PONTOS]\n"
        + "\n".join(f"- {item}" for item in pontos)
        + "\n\n[ACOES]\n"
        + "\n".join(f"- {item}" for item in acoes)
    )



def generate_smart_insights(kpis, top_vendedores_dict, revenue_by_channel_dict, status_dict):
    insights = []
    top_channel = next(iter(revenue_by_channel_dict.items()), None)
    top_vendedor = next(iter(top_vendedores_dict.items()), None)
    second_vendedor = None
    if len(top_vendedores_dict) > 1:
        second_vendedor = list(top_vendedores_dict.items())[1]
    top_status = next(iter(status_dict.items()), None)

    total_channel_revenue = sum(revenue_by_channel_dict.values()) if revenue_by_channel_dict else 0

    if top_channel and total_channel_revenue > 0:
        canal_nome, canal_valor = top_channel
        percentual = (canal_valor / total_channel_revenue) * 100
        if percentual >= 50:
            insights.append(
                f"O canal {canal_nome} concentra {percentual:.1f}% da receita. Há eficiência clara, mas também risco de concentração."
            )
        elif percentual >= 35:
            insights.append(
                f"O canal {canal_nome} lidera com {percentual:.1f}% da receita e deve ser tratado como principal alavanca de escala."
            )

    if top_vendedor:
        vendedor_nome, vendedor_valor = top_vendedor
        if second_vendedor:
            segundo_nome, segundo_valor = second_vendedor
            if segundo_valor > 0 and vendedor_valor >= segundo_valor * 1.4:
                insights.append(
                    f"{vendedor_nome} está muito acima de {segundo_nome} em receita. Existe um padrão comercial que vale replicar no time."
                )
        else:
            insights.append(
                f"{vendedor_nome} lidera sozinho o desempenho comercial no recorte atual e deve servir como referência de execução."
            )

    if kpis["crescimento"] >= 20:
        insights.append(
            f"O crescimento de {kpis['crescimento']:.1f}% indica momento favorável para escalar investimento nos canais mais eficientes."
        )
    elif kpis["crescimento"] <= -5:
        insights.append(
            f"A retração de {abs(kpis['crescimento']):.1f}% exige correção rápida em execução comercial e priorização de canais com melhor retorno."
        )

    if kpis["ticket_medio"] > 0:
        if kpis["ticket_medio"] < 150:
            insights.append(
                f"O ticket médio de {to_br_currency(kpis['ticket_medio'])} está baixo para ganho de rentabilidade. Upsell e cross-sell devem entrar como prioridade."
            )
        elif kpis["ticket_medio"] >= 300:
            insights.append(
                f"O ticket médio de {to_br_currency(kpis['ticket_medio'])} mostra boa captura de valor por venda e permite focar mais em volume qualificado."
            )

    if top_status:
        status_nome, status_qtd = top_status
        if kpis["total_vendas"] > 0:
            status_percentual = (status_qtd / kpis["total_vendas"]) * 100
            if status_percentual >= 45:
                insights.append(
                    f"O status {status_nome} concentra {status_percentual:.1f}% das vendas. Esse ponto merece leitura operacional prioritária."
                )

    return insights[:5]


def build_decision_engine_answer(
    kpis,
    top_vendedores_dict,
    revenue_by_channel_dict,
    status_dict,
    smart_insights,
):
    fallback = build_fallback_sections(
        kpis,
        top_vendedores_dict,
        revenue_by_channel_dict,
        status_dict,
    )

    top_channel = next(iter(revenue_by_channel_dict.items()), None)
    top_vendedor = next(iter(top_vendedores_dict.items()), None)

    resumo = (
        smart_insights[0]
        if smart_insights
        else fallback["resumo"]
    )

    pontos = list(smart_insights[:3]) if smart_insights else list(fallback["pontos"])

    acoes = []
    if top_channel:
        acoes.append(
            safe_sentence(
                f"Direcionar o próximo ciclo de investimento para {top_channel[0]} enquanto o canal mantém liderança comprovada.",
                150,
            )
        )
    if top_vendedor:
        acoes.append(
            safe_sentence(
                f"Replicar a abordagem de {top_vendedor[0]} no restante do time para reduzir dispersão de performance.",
                150,
            )
        )

    if kpis["ticket_medio"] < 150:
        acoes.append("Executar uma frente de upsell e cross-sell para elevar ticket médio sem depender apenas de novo volume.")
    elif kpis["crescimento"] <= 0:
        acoes.append("Revisar rapidamente a alocação comercial e cortar frentes de baixa eficiência para recuperar tração.")
    else:
        acoes.append("Monitorar semanalmente canal líder, vendedor líder e ticket médio para sustentar crescimento com disciplina.")

    while len(pontos) < 3:
        pontos.append(fallback["pontos"][len(pontos)])
    while len(acoes) < 3:
        acoes.append(fallback["acoes"][len(acoes)])

    pontos = [safe_sentence(p, 150) for p in pontos[:3]]
    acoes = [safe_sentence(a, 150) for a in acoes[:3]]

    return (
        "[RESUMO]\n"
        f"{safe_sentence(resumo, 220)}\n\n"
        "[PONTOS]\n"
        + "\n".join(f"- {item}" for item in pontos)
        + "\n\n[ACOES]\n"
        + "\n".join(f"- {item}" for item in acoes)
    )


# ==============================
# ENDPOINTS API
# ==============================
@app.get("/")
def root():
    return {"message": "Backend online com Gemini"}


@app.get("/filters")
def get_filters():
    return {
        "canais": sorted(df["Canal"].dropna().unique().tolist()),
        "status": sorted(df["Status"].dropna().unique().tolist()),
        "vendedores": sorted(df["Vendedor"].dropna().unique().tolist()),
    }


@app.get("/kpis")
def get_kpis(
    canal: str | None = Query(None),
    status: str | None = Query(None),
    vendedor: str | None = Query(None),
):
    df_filtered = apply_filters(df, canal, status, vendedor)
    return get_kpis_from_df(df_filtered)


@app.get("/revenue-series")
def revenue_series(
    canal: str | None = Query(None),
    status: str | None = Query(None),
    vendedor: str | None = Query(None),
):
    df_filtered = apply_filters(df, canal, status, vendedor)
    df_grouped = df_filtered.groupby(df_filtered["Data"].dt.to_period("M"))["Valor (R$)"].sum()
    return [{"date": str(date), "value": float(value)} for date, value in df_grouped.items()]


@app.get("/top-vendedores")
def top_vendedores_endpoint(canal: str = None, status: str = None, vendedor: str = None):
    df_filtered = apply_filters(df, canal, status, vendedor)
    ranking = (
        df_filtered.groupby("Vendedor")["Valor (R$)"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(5)
    )
    ranking.columns = ["vendedor", "receita"]
    return ranking.to_dict(orient="records")


@app.get("/revenue-by-channel")
def revenue_by_channel(canal: str = None, status: str = None, vendedor: str = None):
    df_filtered = apply_filters(df, canal, status, vendedor)
    if df_filtered.empty:
        return []
    ranking = (
        df_filtered.groupby("Canal")["Valor (R$)"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total = ranking["Valor (R$)"].sum()
    result = []
    for _, row in ranking.iterrows():
        result.append({
            "canal": row["Canal"],
            "receita": float(row["Valor (R$)"]),
            "percentual": float((row["Valor (R$)"] / total) * 100) if total else 0,
        })
    return result


@app.get("/forecast")
def forecast(canal: str = None, status: str = None, vendedor: str = None):
    df_filtered = apply_filters(df, canal, status, vendedor)
    if df_filtered.empty:
        return []
    series = df_filtered.groupby("Data")["Valor (R$)"].sum().sort_index()
    values = series.values
    if len(values) < 3:
        return []
    growth_rates = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev > 0:
            growth_rates.append((curr - prev) / prev)
    avg_growth = sum(growth_rates[-3:]) / min(3, len(growth_rates))
    forecast_list = []
    last_value = values[-1]
    for i in range(1, 4):
        last_value = last_value * (1 + avg_growth)
        forecast_list.append({"date": f"F+{i}", "value": float(last_value)})
    return forecast_list

def detect_mode(question: str):
    q = (question or "").lower()

    if any(term in q for term in [
        "estrategia", "estratégia", "investir", "prioridade",
        "direção", "direcao", "decisão", "decisao", "onde focar"
    ]):
        return "CEO"

    if any(term in q for term in [
        "crescer", "crescimento", "escalar", "escala",
        "aumentar vendas", "mais vendas", "expandir"
    ]):
        return "GROWTH"

    if any(term in q for term in [
        "problema", "gargalo", "ineficiencia", "ineficiência",
        "queda", "erro", "baixo desempenho", "melhorar processo"
    ]):
        return "OPERACIONAL"

    return "DEFAULT"

def answer_from_rules(
    question,
    kpis,
    top_vendedores_dict,
    revenue_by_channel_dict,
    status_dict,
):
    q = (question or "").strip().lower()

    if not q:
        return None

    top_vendedor = next(iter(top_vendedores_dict.items()), None)
    top_canal = next(iter(revenue_by_channel_dict.items()), None)
    top_status = next(iter(status_dict.items()), None)

    resumo = ""
    pontos = []
    acoes = []

    # Melhor canal / canal líder
    if any(term in q for term in ["melhor canal", "canal vende mais", "canal lider", "canal líder", "qual canal", "onde investir"]):
        if top_canal:
            canal_nome, canal_valor = top_canal
            resumo = f"{canal_nome} é o principal driver de receita no recorte atual e deve concentrar a prioridade comercial."
            pontos = [
                f"{canal_nome} lidera com {to_br_currency(canal_valor)} em receita.",
                f"O canal líder deve receber foco antes de dispersar esforço em frentes menores.",
                f"A concentração atual revela uma alavanca clara para captura de ganho rápido.",
            ]
            acoes = [
                f"Priorizar {canal_nome} com maior intensidade comercial no curto prazo.",
                f"Reforçar orçamento e execução no canal {canal_nome}.",
                f"Usar o canal líder como referência para replicar práticas nos demais.",
            ]

    # Melhor vendedor / top vendedor
    elif any(term in q for term in ["melhor vendedor", "top vendedor", "quem vende mais", "quem lidera", "vendedor lider", "vendedor líder"]):
        if top_vendedor:
            vendedor_nome, vendedor_valor = top_vendedor
            resumo = f"{vendedor_nome} lidera a performance comercial e deve servir como referência prática para o restante do time."
            pontos = [
                f"{vendedor_nome} lidera with {to_br_currency(vendedor_valor)} no período.",
                f"O desempenho do líder revela um padrão operacional que merece replicação imediata.",
                f"A distância para os demais vendedores ajuda a definir prioridade de coaching.",
            ]
            acoes = [
                f"Mapear o processo comercial de {vendedor_nome}.",
                f"Replicar abordagem, rotina e discurso do vendedor líder no time.",
                f"Criar acompanhamento para elevar consistência dos demais vendedores.",
            ]

    # Ticket médio
    elif any(term in q for term in ["ticket medio", "ticket médio", "aumentar ticket", "elevar ticket"]):
        resumo = f"O ticket médio atual de {to_br_currency(kpis['ticket_medio'])} exige ações para elevar valor por venda sem depender apenas de volume."
        pontos = [
            f"O ticket médio está em {to_br_currency(kpis['ticket_medio'])}.",
            f"Ganho de ticket tende a ser a alavanca mais eficiente para expandir receita com a base atual.",
            f"Upsell e cross-sell devem entrar como prioridade comercial imediata.",
        ]
        acoes = [
            "Testar ofertas de upsell nas vendas em andamento.",
            "Criar pacotes de maior valor para elevar ticket médio.",
            "Treinar o time para capturar receita adicional por transação.",
        ]

    # Crescimento / cenário geral
    elif any(term in q for term in ["crescimento", "cenario", "cenário", "como estamos", "resumo geral", "visao geral", "visão geral"]):
        crescimento = kpis["crescimento"]
        if crescimento > 0:
            resumo = f"O crescimento de {crescimento:.1f}% confirma tração comercial e pede foco em escala com disciplina de execução."
        elif crescimento < 0:
            resumo = f"A queda de {abs(crescimento):.1f}% exige correção rápida na operação comercial e revisão das alavancas principais."
        else:
            resumo = "A operação está estável e precisa de ações claras para retomar aceleração comercial."

        pontos = [
            f"A receita total está em {to_br_currency(kpis['receita_total'])}.",
            f"O volume atual soma {kpis['total_vendas']} vendas no período.",
            f"O ticket médio de {to_br_currency(kpis['ticket_medio'])} ajuda a definir a próxima alavanca de eficiência.",
        ]
        acoes = [
            "Priorizar o principal driver de receita do período.",
            "Ajustar foco comercial com base no canal e vendedor líder.",
            "Usar ticket médio e crescimento como guias da próxima decisão.",
        ]

    # Status dominante / risco operacional
    elif any(term in q for term in ["status", "risco", "gargalo", "ineficiencia", "ineficiência"]):
        if top_status:
            status_nome, status_qtd = top_status
            resumo = f"O status {status_nome} concentra o maior volume e deve orientar a leitura de risco e eficiência operacional."
            pontos = [
                f"{status_nome} concentra {int(status_qtd)} vendas no recorte atual.",
                f"O status dominante revela onde a operação está mais concentrada.",
                f"Entender esse bloco é essencial para reduzir atrito e melhorar conversão.",
            ]
            acoes = [
                f"Analisar a etapa vinculada ao status {status_nome}.",
                "Mapear gargalos operacionais que travam avanço de receita.",
                "Corrigir primeiro o ponto de maior concentração operacional.",
            ]

    if not resumo:
        return None

    return (
        "[RESUMO]\n"
        f"{safe_sentence(resumo, 220)}\n\n"
        "[PONTOS]\n"
        + "\n".join(f"- {safe_sentence(p, 150)}" for p in pontos[:3])
        + "\n\n[ACOES]\n"
        + "\n".join(f"- {safe_sentence(a, 150)}" for a in acoes[:3])
    )

# ==============================
# IA INTEGRADA COM FILTROS
# ==============================
@app.post("/ask-ai")
def ask_ai(payload: dict = Body(...)):
    question = payload.get("question", "").strip()
    filters = payload.get("filters", {})

    canal = filters.get("canal")
    status = filters.get("status")
    vendedor = filters.get("vendedor")

    df_filtered = apply_filters(df, canal, status, vendedor)

    if df_filtered.empty:
        return {
            "answer": (
                "[RESUMO]\n"
                "Nenhum dado disponível para os filtros selecionados.\n\n"
                "[PONTOS]\n"
                "- Nenhum registro encontrado no recorte atual.\n"
                "- Revise os filtros aplicados antes de interpretar o cenário.\n"
                "- Amplie o escopo para gerar uma leitura executiva útil.\n\n"
                "[ACOES]\n"
                "- Remover filtros muito restritivos.\n"
                "- Testar outro canal, status ou vendedor.\n"
                "- Reprocessar a consulta com mais dados disponíveis."
            )
        }

    kpis = get_kpis_from_df(df_filtered)

    top_vendedores_dict = (
        df_filtered.groupby("Vendedor")["Valor (R$)"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    revenue_by_channel_dict = (
        df_filtered.groupby("Canal")["Valor (R$)"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    status_dict = (
        df_filtered.groupby("Status")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # =========================
    # 🔥 MODO HÍBRIDO (REGRAS LOCAIS)
    # =========================
    rule_answer = answer_from_rules(
        question,
        kpis,
        top_vendedores_dict,
        revenue_by_channel_dict,
        status_dict,
    )

    if rule_answer:
        return {"answer": rule_answer}

    # =========================
    # 🔥 MOTOR DE DECISÃO (ANTES DA IA)
    # =========================
    smart_insights = generate_smart_insights(
        kpis,
        top_vendedores_dict,
        revenue_by_channel_dict,
        status_dict,
    )

    if not question:
        return {
            "answer": build_decision_engine_answer(
                kpis,
                top_vendedores_dict,
                revenue_by_channel_dict,
                status_dict,
                smart_insights,
            )
        }

    # =========================
    # IA (SÓ SE NÃO FOR PERGUNTA OBJETIVA)
    # =========================

    top_vendedores_text = (
        "\n".join([f"- {nome}: {to_br_currency(valor)}" for nome, valor in top_vendedores_dict.items()])
        if top_vendedores_dict
        else "- Sem dados suficientes no filtro atual"
    )

    revenue_by_channel_text = (
        "\n".join([f"- {canal_nome}: {to_br_currency(valor)}" for canal_nome, valor in revenue_by_channel_dict.items()])
        if revenue_by_channel_dict
        else "- Sem dados por canal no filtro atual"
    )

    status_text = (
        "\n".join([f"- {status_nome}: {int(qtd)} vendas" for status_nome, qtd in status_dict.items()])
        if status_dict
        else "- Sem dados de status no filtro atual"
    )

    mode = detect_mode(question)

    mode_instruction = ""

    if mode == "CEO":
        mode_instruction = """
Responda como um executivo.
Foque em direção estratégica, decisão e alocação de recursos.
Seja direto, assertivo e orientado a impacto.
"""

    elif mode == "GROWTH":
        mode_instruction = """
Responda como especialista em crescimento.
Foque em escala, aumento de receita e alavancas de crescimento.
Traga ideias práticas para crescer mais rápido.
"""

    elif mode == "OPERACIONAL":
        mode_instruction = """
Responda como gestor operacional.
Foque em eficiência, gargalos e melhoria de processo.
Aponte problemas e correções diretas.
"""

    prompt = f"""
Você é um consultor sênior de revenue analytics em um dashboard SaaS premium.

{mode_instruction}

Dados do período atual:
- Receita total: {to_br_currency(kpis["receita_total"])}
- Total de vendas: {kpis["total_vendas"]}
- Ticket médio: {to_br_currency(kpis["ticket_medio"])}
- Crescimento: {kpis["crescimento"]:.1f}%

Top vendedores:
{top_vendedores_text}

Receita por canal:
{revenue_by_channel_text}

Distribuição por status:
{status_text}

Insights determinísticos do motor de decisão:
{chr(10).join(f"- {item}" for item in smart_insights) if smart_insights else "- Nenhum insight determinístico relevante foi identificado."}

Pergunta do usuário:
{question if question else "Analise os dados atuais com foco em decisão."}

Formato obrigatório:

[RESUMO]
2 frases executivas com leitura + decisão.

[PONTOS]
- Insight estratégico direto.
- Insight estratégico direto.
- Insight estratégico direto.

[ACOES]
- Ação prática e priorizada.
- Ação prática e priorizada.
- Ação prática e priorizada.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 450,
            },
        )

        answer = response.text.strip() if response.text else ""

        # =========================
        # 🔥 NORMALIZAÇÃO PREMIUM
        # =========================
        replacements = {
            "indica": "confirma",
            "sugere": "exige",
            "aponta": "revela",
            "merece atenção": "é crítico",
            "há espaço para": "exige ação para",
            "pode melhorar": "precisa melhorar",
            "pode aumentar": "deve aumentar",
        }

        for k, v in replacements.items():
            answer = re.sub(rf"\b{k}\b", v, answer, flags=re.IGNORECASE)

        return {"answer": answer}

    except Exception as e:
        error_text = str(e)

    if "RESOURCE_EXHAUSTED" in error_text:
        fallback = build_fallback_sections(
            kpis,
            top_vendedores_dict,
            revenue_by_channel_dict,
            status_dict
        )

        return {
            "answer": (
                "[RESUMO]\n"
                f"{fallback['resumo']}\n\n"
                "[PONTOS]\n"
                + "\n".join([f"- {p}" for p in fallback["pontos"]]) + "\n\n"
                "[ACOES]\n"
                + "\n".join([f"- {a}" for a in fallback["acoes"]])
            )
        }

    return {"answer": f"Erro Gemini: {error_text}"}