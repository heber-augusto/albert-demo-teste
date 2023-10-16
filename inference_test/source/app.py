from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flasgger import Swagger
from code import perform_inference, load_generative_model
import os

app = Flask(__name__)
api = Api(app)
swagger = Swagger(app)

# Variável global para armazenar o modelo carregado
generative_model = None

class Inference(Resource):
    def post(self):
        """
        Execute a inferência com um modelo de machine learning.
        ---
        tags:
          - inferência
        parameters:
          - name: input_content
            in: formData
            type: string
            required: true
            description: Conteúdo de entrada para inferência
        responses:
          200:
            description: Resultado da inferência
        """
        # Execute a inferência
        input_content = request.form['input_content']
        app.logger.warning(f'modelo em uso: {generative_model}')
        result = perform_inference(input_content, generative_model)

        return result


def initialize_app():
    global generative_model
    app.logger.warning(f'carregando modelo')
    if generative_model is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        # Realize o carregamento do modelo, por exemplo, usando seu código atual
        generative_model = load_generative_model(model_dir)
        app.logger.warning(f'modelo carregado com sucesso {generative_model}')


# Rota para a página inicial que redireciona para a documentação do Swagger
@app.route('/')
def index():
    return redirect('/apidocs')


api.add_resource(Inference, '/inference')

app.logger.warning(f'carregando app')
initialize_app()

if __name__ == '__main__':
    app.run(debug=True)
