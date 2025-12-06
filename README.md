# API-FLASK_IA

------------------------------------------------------------
1. Estrutura do Projeto
------------------------------------------------------------

/projeto
├── modelo_desempenho_bertimbau/
├── api_flask.py
├── teste_local.py
├── treinar_modelo.py
├── desempenho_funcionarios.csv
└── README.md

------------------------------------------------------------
2. Endpoint /avaliar
------------------------------------------------------------

Requisição (POST):

{
  "resposta_aberta": "Texto do colaborador...",
  "pergunta_aberta": "Pergunta aplicada",
  "total_perguntas": 10,
  "acertos": 7,
  "id_usuario": "123",
  "id_conteudo": "abc"
}

------------------------------------------------------------
3. Processamento da API
------------------------------------------------------------

1. Aplicação do modelo BERTimbau na resposta aberta.
2. Cálculo do score do quiz: quiz_score = acertos / total_perguntas
3. Combinação dos dois resultados:
   score_final = 0.6 * score_bert + 0.4 * quiz_score
4. Classificação final por faixas:

   < 0.33  = baixo
   < 0.66  = medio
   >= 0.66 = alto

5. Registro dos dados no Supabase via requisição REST.

------------------------------------------------------------
4. Resposta da API
------------------------------------------------------------

{
  "classe_bert": 2,
  "classe_bert_label": "alto",
  "score_bert": 0.8234,
  "quiz_score": 0.7,
  "score_final": 0.7694,
  "classe_final": 2
}

------------------------------------------------------------
5. Integração com Supabase
------------------------------------------------------------

Método: POST
Endpoint:
https://SEU_URL_SUPABASE/rest/v1/avaliacoes_desempenho

Cabeçalhos obrigatórios:
apikey: SUA_CHAVE_SUPABASE
Authorization: Bearer SUA_CHAVE_SUPABASE
Content-Type: application/json

Campos enviados:
id_usuario
id_conteudo
classe_final
score_final
classe_bert
score_bert
quiz_score
acertos
total_perguntas
pergunta_aberta
resposta_aberta
timestamp (opcional)

------------------------------------------------------------
6. Estrutura Recomendada de Tabela no Supabase
------------------------------------------------------------

CREATE TABLE avaliacoes_desempenho (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  id_usuario text NOT NULL,
  id_conteudo text NOT NULL,
  classe_final integer NOT NULL,
  score_final numeric NOT NULL,
  classe_bert integer NOT NULL,
  score_bert numeric NOT NULL,
  quiz_score numeric NOT NULL,
  acertos integer NOT NULL,
  total_perguntas integer NOT NULL,
  pergunta_aberta text,
  resposta_aberta text,
  criado_em timestamp DEFAULT now()
);

------------------------------------------------------------
7. Variáveis de Ambiente
------------------------------------------------------------

SUPABASE_URL="https://SEU_URL.supabase.co"
SUPABASE_KEY="SUA_CHAVE_API"
PORT=5000

Opcional:
MODEL_DIR="./modelo_desempenho_bertimbau"

Salvar em arquivo .env ou configuração equivalente.

------------------------------------------------------------
8. Teste Local (teste_local.py)
------------------------------------------------------------

Comando:

python teste_local.py

Saída esperada:

Classe prevista: alto (id = 2)
Probabilidades:
  0 (baixo): 0.123
  1 (medio): 0.044
  2 (alto): 0.833

------------------------------------------------------------
9. Treinamento do Modelo (treinar_modelo.py)
------------------------------------------------------------

Formato do CSV:

texto,label
"Excelente comunicação e proatividade",alto
"Cumpre tarefas básicas",medio
"Falta comprometimento",baixo

Executar:

python treinar_modelo.py

Saída:
Modelo salvo no diretório:
./modelo_desempenho_bertimbau

------------------------------------------------------------
10. Classes
------------------------------------------------------------

0 = baixo
1 = medio
2 = alto

------------------------------------------------------------
11. Fórmula do Score Final
------------------------------------------------------------

score_final = 0.6 * score_bert + 0.4 * quiz_score

------------------------------------------------------------
12. Dependências
------------------------------------------------------------

IA:
torch
transformers
datasets
scikit-learn
pandas
numpy

API:
flask
requests
python-dotenv (opcional)

Supabase (REST): não é necessária lib oficial

------------------------------------------------------------
13. Execução Local
------------------------------------------------------------

1. Instalar dependências:
pip install -r requirements.txt

2. Executar a API:
python api_flask.py

3. Testar endpoint:
http://localhost:5000/avaliar

------------------------------------------------------------
14. Deploy
------------------------------------------------------------

Ambientes recomendados:

- Render (deploy rápido Flask + Python)
- Railway (banco e API no mesmo projeto)
- Google Cloud Run
- VPS com Ubuntu 22

Checklist de Deploy:

1. Exportar variáveis de ambiente SUPABASE_URL e SUPABASE_KEY
2. Ativar CORS no Flask se necessário
3. Liberar rota POST no Supabase com RLS configurado
