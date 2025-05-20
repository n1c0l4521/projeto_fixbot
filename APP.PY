from flask import Flask, request, jsonify, render_template
import ast
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re

app = Flask(__name__)

# Carrega o modelo e o tokenizer uma vez
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base')
model.eval()  # Modo de avaliação para desativar dropout etc.

# Página inicial
@app.route('/')
def home():
    return render_template("index.html")

# Função de pré-processamento corrigida
def preprocess_code(code):
    # Mantém a indentação e quebra de linha, removendo apenas espaços em branco à direita
    lines = code.strip().splitlines()
    cleaned_lines = [line.rstrip() for line in lines]
    return '\n'.join(cleaned_lines)

# Verificação de sintaxe
def check_syntax(code):
    try:
        ast.parse(code)
        return "Código sintaticamente correto."
    except SyntaxError as e:
        mensagem = traduzir_erro(str(e))
        return f"Erro de sintaxe na linha {e.lineno}: {mensagem}"

# Tradução de mensagens de erro
def traduzir_erro(erro):
    traducoes = {
        "expected an indented block": "esperado um bloco indentado",
        "was never closed": "nunca foi fechado",
        "unexpected EOF while parsing": "fim inesperado do arquivo durante a análise",
        "invalid syntax": "sintaxe inválida",
        "unexpected indent": "indentação inesperada",
        "unindent does not match any outer indentation level": "desindentação não corresponde a nenhum nível externo",
        "name '": "nome '",
        "is not defined": "não está definido",
    }
    for original, traducao in traducoes.items():
        erro = erro.replace(original, traducao)
    return erro

# Detecção de erro lógico (com base no modelo CodeBERT)
def detect_code_error(code):
    try:
        # Pré-processar e tokenizar o código
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        
        # Fazer a inferência com o modelo
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extrair as logits e calcular as probabilidades
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
        # Pegar a classe com a maior probabilidade
        confidence, prediction = torch.max(probabilities, dim=1)
        
        # Interpretar a previsão
        label = "correto" if prediction.item() == 0 else "com erro"
        porcentagem = confidence.item() * 100
        
        # Retornar a mensagem de erro lógico com a porcentagem de confiança
        if label == "correto":
            return f"Código logicamente correto (Confiança: {porcentagem:.2f}%)"
        else:
            return f"Erro lógico detectado (Confiança: {porcentagem:.2f}%)"
    except Exception as e:
        return f"Erro lógico detectado: {str(e)} (Confiança: 100%)"

# Rota de verificação
@app.route("/process_code", methods=["POST"])
def process_code():
    try:
        data = request.get_json(force=True)
        code = data.get("code", "")

        if not code.strip():
            return jsonify({"result": "Nenhum código fornecido."})

        # Pré-processamento preservando indentação
        code = preprocess_code(code)

        # Verificações
        syntax_message = check_syntax(code)
        logic_message = detect_code_error(code)

        # Resultado final
        result = (
            f"Verificação de Sintaxe: {syntax_message}\n"
            f"Verificação Lógica: {logic_message}"
        )
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": f"Erro no servidor: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
