import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "./modelo_desempenho_bertimbau"


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


if hasattr(model.config, "id2label") and model.config.id2label:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
else:
    id2label = {
        0: "baixo",
        1: "medio",
        2: "alto"
    }

def analisar_texto(texto: str):

    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits


    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()


    pred_id = int(torch.argmax(logits, dim=-1).cpu().item())
    rotulo = id2label.get(pred_id, str(pred_id))

    return pred_id, rotulo, probs


if __name__ == "__main__":


    print("=== Teste do modelo de desempenho ===")
    print("Digite 'sair' a qualquer momento para encerrar.\n")

    while True:
        pergunta = input("Pergunta (opcional, pode deixar em branco): ")
        if pergunta.lower() == "sair":
            break

        resposta = input("Resposta do funcionário: ")
        if resposta.lower() == "sair":
            break

        if pergunta.strip():
            texto_modelo = f"Pergunta: {pergunta}\nResposta: {resposta}"
        else:
            texto_modelo = resposta

        pred_id, rotulo, probs = analisar_texto(texto_modelo)

        print("\n➡ Resultado da análise:")
        print(f"  Classe prevista: {rotulo} (id={pred_id})")
        print("  Probabilidades por classe:")
        for i, p in enumerate(probs):
            nome_classe = id2label.get(i, str(i))
            print(f"    {i} ({nome_classe}): {p:.3f}")
        print("\n" + "-" * 50 + "\n")
