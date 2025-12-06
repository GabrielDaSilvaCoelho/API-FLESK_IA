import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import requests

# ============================================================
# 1) CARREGAR MODELO LOCAL
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo_desempenho_bertimbau")

print("MODEL_DIR:", MODEL_DIR)
if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"Pasta do modelo não encontrada em: {MODEL_DIR}")

print("Carregando modelo e tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

device = torch.device("cpu")
model.to(device)
print("Usando device:", device)

app = Flask(__name__)

LABEL2TEXT = {
    0: "Baixo desempenho",
    1: "Desempenho mediano",
    2: "Alto desempenho"
}

# ============================================================
# 2) CONFIGURAÇÃO DO SUPABASE
# ============================================================
SUPABASE_URL = "https://pbpkxbkwfpznkkuwcxjl.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBicGt4Ymt3ZnB6bmtrdXdjeGpsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTMzMDMzNiwiZXhwIjoyMDc2OTA2MzM2fQ.BcFYahPvE1qMQ9xqBdXB2AE7joH41v-paA8nIBWnS1I"

TABELA_AVALIACOES = "avaliacoes_desempenho"


def salvar_avaliacao_supabase(payload: dict):
    """
    Envia a avaliação para o Supabase via REST.
    """
    try:
        url = f"{SUPABASE_URL}/rest/v1/{TABELA_AVALIACOES}"
        headers = {
            "apikey": SUPABASE_API_KEY,
            "Authorization": f"Bearer {SUPABASE_API_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=10)

        if not resp.ok:
            print("Erro ao salvar avaliação no Supabase:",
                  resp.status_code, resp.text)
        else:
            print("Avaliação salva no Supabase. Status:", resp.status_code)

    except Exception as e:
        print("Exceção ao salvar avaliação no Supabase:", e)


# ============================================================
# 3) ENDPOINT /avaliar (Chamado pelo app Android)
# ============================================================
@app.route("/avaliar", methods=["POST"])
def avaliar():
    data = request.get_json(force=True)

    resposta_aberta = data.get("resposta_aberta", "").strip()
    pergunta_aberta = data.get("pergunta_aberta", "")
    total_perguntas = int(data.get("total_perguntas", 0))
    acertos = int(data.get("acertos", 0))
    id_usuario = data.get("id_usuario")
    id_conteudo = data.get("id_conteudo")

    if not resposta_aberta:
        return jsonify({"erro": "resposta_aberta vazia"}), 400

    if total_perguntas <= 0:
        total_perguntas = 1

    # ---------------- 1) IA BERT ----------------
    inputs = tokenizer(
        resposta_aberta,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0].tolist()

    classe_bert = int(torch.argmax(logits, dim=-1).item())
    score_bert = float(probs[classe_bert])      # confiança da classe escolhida (0–1)
    classe_bert_label = LABEL2TEXT.get(classe_bert, "Desconhecido")

    # ---------------- 2) Score do quiz ----------------
    quiz_score = acertos / total_perguntas      # 0–1

    # ---------------- 3) Score final combinado ----------------
    peso_ia = 0.6
    peso_quiz = 0.4

    score_final = peso_ia * score_bert + peso_quiz * quiz_score  # 0–1

    # ---------------- 4) Classe final de desempenho ----------------
    if score_final < 0.33:
        classe_final = 0
    elif score_final < 0.66:
        classe_final = 1
    else:
        classe_final = 2

    # Resposta para o ANDROID
    resposta = {
        "id_usuario": id_usuario,
        "id_conteudo": id_conteudo,

        # IA (texto)
        "classe_bert": classe_bert,
        "classe_bert_label": classe_bert_label,
        "score_bert": round(score_bert, 4),

        # Quiz
        "total_perguntas": total_perguntas,
        "acertos": acertos,
        "quiz_score": round(quiz_score, 4),

        # Final
        "score_final": round(score_final, 4),
        "classe_final": classe_final
    }

    # ---------------- 5) Salvar no Supabase (ALINHADO COM O BANCO) ----------------
    supabase_payload = {
        "id_usuario": id_usuario,
        "id_conteudo": id_conteudo,
        "classe_final": classe_final,
        "score_final": score_final,
        "classe_bert_label": classe_bert_label,
        "classe_bert": classe_bert,
        "score_bert": score_bert,
        "total_perguntas": total_perguntas,
        "acertos": acertos,
        "quiz_score": quiz_score,
        "pergunta_aberta": pergunta_aberta,
        "resposta_aberta": resposta_aberta
    }

    salvar_avaliacao_supabase(supabase_payload)

    return jsonify(resposta), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
