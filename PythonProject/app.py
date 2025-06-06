from flask import Flask, render_template
from flasgger import Swagger
from ai_routes import ai_bp
from dotenv import load_dotenv
from prometheus_flask_exporter import PrometheusMetrics

load_dotenv()

app = Flask(__name__)
Swagger(app)                         # 📄 Swagger UI at /apidocs
metrics = PrometheusMetrics(app)    # ✅ Initialize after app is defined

app.register_blueprint(ai_bp)

@app.route('/')
def index():
    return render_template("index.html", project_name="AI Assistant")

if __name__ == '__main__':
    app.run(debug=True)
