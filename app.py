import io
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, abort, jsonify, Response
from werkzeug.utils import secure_filename
from analysis_utils import (
    read_csv_in_chunks,
    analyze_dataframe,
    remove_missing_from_df, 
    remove_duplicates_from_df, 
    fill_missing_values,
    process_chunk_with_dates
)
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from flask_mail import Mail, Message
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import secrets
import re
from werkzeug.security import check_password_hash
from flask_migrate import Migrate
from statistic_utils import (
    perform_statistical_analysis,
    remove_outliers,
    get_column_statistics,
    create_histogram_plot,
    create_bar_plot,
    create_boxplot
)
from ai_agent import generate_agent_report


app = Flask(__name__)

current_df = None
# Журнал предобработки данных: список шагов очистки и трансформаций
preprocessing_log = []
last_agent_report = None
last_agent_target_column = None

load_dotenv()

app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@postgres:5432/vkr_db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.yandex.ru')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 465))
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

# Инициализация
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

migrate = Migrate(app, db)

# Модель ролей
class Role(db.Model):
    __tablename__ = 'role'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(200))

    def __repr__(self):
        return f'<Role {self.name}>'

# Модель пользователя
class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    confirmed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    password_reset_token = db.Column(db.String(100), unique=True)
    password_reset_expires = db.Column(db.DateTime)
    role_id = db.Column(db.Integer, db.ForeignKey('role.id'), nullable=True, default=1)  # По умолчанию роль 'user'
    role = db.relationship('Role', backref='users')

    def __repr__(self):
        return f'<User {self.email}>'

    def has_role(self, role_name):
        return self.role.name == role_name

# Функция для проверки сложности пароля
def is_password_complex(password):
    if len(password) < 8:
        return False, "Пароль должен содержать минимум 8 символов"
    if not re.search(r"[A-Za-z]", password):
        return False, "Пароль должен содержать буквы"
    if not re.search(r"\d", password):
        return False, "Пароль должен содержать цифры"
    if not re.search(r"[ !@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        return False, "Пароль должен содержать специальные символы"
    return True, "Пароль соответствует требованиям"

# Создание ролей при первом запуске
def create_roles():
    with app.app_context():
        if not Role.query.filter_by(name='admin').first():
            admin_role = Role(name='admin', description='Администратор с полными правами')
            db.session.add(admin_role)
        
        if not Role.query.filter_by(name='user').first():
            user_role = Role(name='user', description='Обычный пользователь')
            db.session.add(user_role)
        
        db.session.commit()

# Создаем таблицы и роли
with app.app_context():
    db.create_all()
    create_roles()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({'status': 'error', 'message': 'Все поля обязательны'}), 400
        
        # Проверка сложности пароля
        is_complex, message = is_password_complex(password)
        if not is_complex:
            return jsonify({'status': 'error', 'message': message}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'status': 'error', 'message': 'Email уже используется'}), 400
        
        # По умолчанию назначаем роль 'user'
        user_role = Role.query.filter_by(name='user').first()
        
        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role_id=user_role.id
        )
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return jsonify({'status': 'success', 'message': 'Регистрация успешна'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'status': 'error', 'message': 'Email и пароль обязательны'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'status': 'error', 'message': 'Неверные учетные данные'}), 401
        
        login_user(user)
        return jsonify({'status': 'success', 'message': 'Вход выполнен'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def send_verification_email(email, code):
    try:
        msg = Message(
            "Код подтверждения",
            recipients=[email],
            body=f"Ваш код подтверждения: {code}"
        )
        mail.send(msg)
        print(f"Email sent to {email} with code: {code}")  # Для отладки
        return True
    except Exception as e:
        print(f"Email send error: {str(e)}")  # Для отладки
        return False

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('upload_form'))

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        email = request.json.get('email')
        user = User.query.filter_by(email=email).first()
        
        if not user:
            return jsonify({'status': 'error', 'message': 'Пользователь с таким email не найден'}), 404
        
        # Генерируем токен и срок его действия
        token = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(hours=1)
        
        user.password_reset_token = token
        user.password_reset_expires = expires
        db.session.commit()
        
        # Отправляем email с ссылкой для сброса
        reset_link = f"{request.host_url}reset-password?token={token}"
        msg = Message(
            "Восстановление пароля",
            recipients=[email],
            body=f"Для сброса пароля перейдите по ссылке: {reset_link}\nСсылка действительна 1 час.",
            html=f"""
            <h2>Восстановление пароля</h2>
            <p>Для сброса пароля нажмите на кнопку ниже:</p>
            <a href="{reset_link}" style="background: #4f46e5; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                Сбросить пароль
            </a>
            <p>Ссылка действительна 1 час.</p>
            """
        )
        mail.send(msg)
        
        return jsonify({
            'status': 'success',
            'message': 'На ваш email отправлена инструкция по восстановлению пароля'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'GET':
        # Показываем форму для сброса пароля
        token = request.args.get('token')
        user = User.query.filter_by(password_reset_token=token).first()
        
        if not user or user.password_reset_expires < datetime.utcnow():
            return render_template('invalid_token.html')  # Создайте этот шаблон
        
        return render_template('reset_password.html', token=token)
    
    elif request.method == 'POST':
        # Обрабатываем отправку формы
        token = request.json.get('token')
        new_password = request.json.get('new_password')
        
        user = User.query.filter_by(password_reset_token=token).first()
        
        if not user or user.password_reset_expires < datetime.utcnow():
            return jsonify({'status': 'error', 'message': 'Недействительная или просроченная ссылка'}), 400
        
        user.password_hash = generate_password_hash(new_password)
        user.password_reset_token = None
        user.password_reset_expires = None
        db.session.commit()
        
        return jsonify({
            'status': 'success', 
            'message': 'Пароль успешно изменен',
            'redirect': url_for('upload_form')  # Перенаправление на главную
        })

# Защищенные маршруты --------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    global current_df, preprocessing_log

    if "file" not in request.files:
        flash("Файл не найден в запросе.")
        return redirect(url_for("upload_form"))

    file = request.files["file"]

    if file.filename == "":
        flash("Файл не выбран.")
        return redirect(url_for("upload_form"))

    if not file.filename.lower().endswith('.csv'):
        flash("Поддерживаются только CSV файлы.")
        return redirect(url_for("upload_form"))

    filename = secure_filename(file.filename)

    try:
        # Чтение и обработка чанками с преобразованием дат
        chunks = read_csv_in_chunks(file)
        processed_chunks = [process_chunk_with_dates(chunk) for chunk in chunks]
        full_df = pd.concat(processed_chunks)
        current_df = full_df.copy()

        # Сбрасываем журнал предобработки для нового файла
        preprocessing_log = []

        # Анализ данных
        analysis_results = analyze_dataframe(full_df)
        
        # Добавляем информацию о преобразованных колонках
        date_columns = [col for col in full_df.columns 
                       if pd.api.types.is_datetime64_any_dtype(full_df[col])]
        analysis_results['date_columns'] = date_columns

        # Добавляем в журнал шаг начальной загрузки и анализа
        try:
            total_cells = full_df.shape[0] * full_df.shape[1] if full_df.shape[0] and full_df.shape[1] else 0
            preprocessing_log.append({
                "step": "initial_load",
                "function": "read_csv_in_chunks+process_chunk_with_dates+analyze_dataframe",
                "rows": int(full_df.shape[0]),
                "columns": int(full_df.shape[1]),
                "missing_per_column": {col: float(full_df[col].isna().mean()) for col in full_df.columns},
                "total_missing_fraction": float(full_df.isna().sum().sum() / total_cells) if total_cells else 0.0,
                "total_duplicates": int(full_df.duplicated().sum())
            })
        except Exception:
            # Журнал не должен ломать основной сценарий
            pass

        return render_template("upload.html",
                             filename=filename,
                             **analysis_results)

    except Exception as e:
        flash(f"Ошибка обработки файла: {str(e)}")
        return redirect(url_for("upload_form"))


@app.route("/", methods=["GET"])
def upload_form():
    return render_template("upload_form.html")


# Stream API эндпоинт для потоковой выдачи данных
@app.route('/stream-data')
def stream_data():
    global current_df
    
    if current_df is None:
        abort(404, description="Нет данных для потоковой выдачи")

    def generate():
        # Заголовок CSV
        yield current_df.head(0).to_csv(index=False)
        
        # Разбиваем DataFrame на чанки
        for chunk in np.array_split(current_df, 10):  # 10 чанков
            yield chunk.to_csv(index=False, header=False)

    return Response(
        generate(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=streamed_data.csv',
            'Transfer-Encoding': 'chunked'
        }
    )



@app.route("/download", methods=["GET"])
def download_file():
    """
    Формирует CSV файл из текущего DataFrame и отправляет его на скачивание.
    """
    global current_df

    if current_df is None:
        abort(404, description="Нет загруженного файла для скачивания")

    # Преобразуем DataFrame в CSV
    output = io.StringIO()
    current_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                 mimetype="text/csv",
                 as_attachment=True,
                 download_name="modified_file.csv")



@app.route("/remove-missing", methods=["POST"])
def remove_missing():
    global current_df, preprocessing_log

    if current_df is None:
        return jsonify({"message": "Нет загруженного файла для удаления пропусков"}), 404

    try:
        # Считаем статистику до операции
        rows_before = int(current_df.shape[0])
        missing_before = int(current_df.isnull().sum().sum())
        missing_frac_before = current_df.isna().mean().to_dict()

        # Функция remove_missing_from_df возвращает (df, success_message)
        current_df, success_message = remove_missing_from_df(current_df)

        # Статистика после операции
        rows_after = int(current_df.shape[0])
        missing_after = int(current_df.isnull().sum().sum())
        missing_frac_after = current_df.isna().mean().to_dict()

        # Добавляем шаг в журнал предобработки
        try:
            preprocessing_log.append({
                "step": "remove_missing",
                "function": "remove_missing_from_df",
                "rows_before": rows_before,
                "rows_after": rows_after,
                "removed_rows": rows_before - rows_after,
                "total_missing_before": missing_before,
                "total_missing_after": missing_after,
                "missing_fraction_per_column_before": {k: float(v) for k, v in missing_frac_before.items()},
                "missing_fraction_per_column_after": {k: float(v) for k, v in missing_frac_after.items()},
                "user_message": success_message
            })
        except Exception:
            pass
    except Exception as e:
        print(f"Ошибка при удалении пропусков: {e}")
        success_message = "Ошибка при удалении пропусков"

    return jsonify({"message": success_message})


@app.route("/remove-duplicates", methods=["POST"])
def remove_duplicates():
    global current_df, preprocessing_log

    if current_df is None:
        return jsonify({"message": "Нет загруженного файла для удаления пропусков"}), 404

    try:
        # Статистика до операции
        rows_before = int(current_df.shape[0])
        duplicates_before = int(current_df.duplicated().sum())

        # Функция remove_duplicates_from_df возвращает (df, success_message)
        current_df, success_message = remove_duplicates_from_df(current_df)

        # Статистика после операции
        rows_after = int(current_df.shape[0])
        duplicates_after = int(current_df.duplicated().sum())

        try:
            preprocessing_log.append({
                "step": "remove_duplicates",
                "function": "remove_duplicates_from_df",
                "rows_before": rows_before,
                "rows_after": rows_after,
                "removed_rows": rows_before - rows_after,
                "duplicates_before": duplicates_before,
                "duplicates_after": duplicates_after,
                "user_message": success_message
            })
        except Exception:
            pass
    except Exception as e:
        # Можно залогировать ошибку
        print(f"Ошибка при удалении пропусков: {e}")
        success_message = "Ошибка при удалении пропусков"

    return jsonify({"message": success_message})


@app.route('/fill_missing', methods=['POST'])
def fill_missing():
    global current_df, preprocessing_log
    data = request.get_json()
    categorical_method = data.get('categorical')
    quantitative_method = data.get('quantitative')
    n_neighbors = 5

    try:
        # Статистика до импутации
        rows_before = int(current_df.shape[0])
        missing_before = int(current_df.isnull().sum().sum())
        missing_frac_before = current_df.isna().mean().to_dict()

        # Вызываем функцию заполнения пропусков
        filled_df = fill_missing_values(current_df, categorical_method, quantitative_method, n_neighbors)
        current_df = filled_df

        # Статистика после импутации
        rows_after = int(current_df.shape[0])
        missing_after = int(current_df.isnull().sum().sum())
        missing_frac_after = current_df.isna().mean().to_dict()

        try:
            preprocessing_log.append({
                "step": "fill_missing",
                "function": "fill_missing_values",
                "rows_before": rows_before,
                "rows_after": rows_after,
                "total_missing_before": missing_before,
                "total_missing_after": missing_after,
                "missing_fraction_per_column_before": {k: float(v) for k, v in missing_frac_before.items()},
                "missing_fraction_per_column_after": {k: float(v) for k, v in missing_frac_after.items()},
                "categorical_method": categorical_method,
                "quantitative_method": quantitative_method,
                "n_neighbors": n_neighbors
            })
        except Exception:
            pass
        
        return jsonify({
            "status": "success",
            "message": "Заполнение пропусков выполнено успешно"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Ошибка при заполнении пропусков: {str(e)}"
        })
    
#------------NEW
@app.route("/statistical-analysis", methods=["GET", "POST"])
def statistical_analysis():
    global current_df
    
    if current_df is None:
        flash("Сначала загрузите файл для анализа")
        return redirect(url_for("upload_form"))
    
    target_column = None
    analysis_results = None
    
    if request.method == "POST":
        target_column = request.form.get("target_column")
        analysis_results = perform_statistical_analysis(current_df, target_column)
    
    # Получаем список столбцов для выбора целевой переменной
    numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
    
    return render_template("statistical_analysis.html",
                         df_columns=current_df.columns.tolist(),
                         numeric_columns=numeric_columns,
                         target_column=target_column,
                         results=analysis_results)


@app.route('/final-report', methods=['POST'])
def final_report():
    global current_df, preprocessing_log, last_agent_report, last_agent_target_column

    if current_df is None:
        flash("Сначала загрузите файл для анализа")
        return redirect(url_for('upload_form'))

    target_column = request.form.get('target_column') or None
    # toggle to use Deepseek (checkbox named use_deepseek)
    use_deepseek = request.form.get('use_deepseek') == 'on' or os.getenv('USE_DEEPSEEK', 'true').lower() == 'true'

    try:
        import markdown
        report = generate_agent_report(
            current_df,
            target_column,
            use_deepseek=use_deepseek,
            preprocessing_log=preprocessing_log
        )
        
        # Конвертируем markdown в HTML, если есть отчет от DeepSeek
        if report.get('deepseek_report'):
            try:
                # Расширения для лучшей обработки markdown
                # extra - добавляет таблицы, аббревиатуры и другие функции
                # nl2br - конвертирует переносы строк в <br>
                extensions = ['extra', 'nl2br']
                md = markdown.Markdown(extensions=extensions)
                report['deepseek_report_html'] = md.convert(report['deepseek_report'])
            except Exception as e:
                # Если расширения не работают, используем базовый markdown
                print(f"Ошибка при конвертации markdown: {e}")
                md = markdown.Markdown()
                report['deepseek_report_html'] = md.convert(report['deepseek_report'])
        else:
            report['deepseek_report_html'] = None
            
    except Exception as e:
        import traceback
        error_msg = f"Ошибка при генерации отчёта: {str(e)}\n{traceback.format_exc()}"
        flash(error_msg)
        print(error_msg)  # Логируем для отладки
        return redirect(url_for('statistical_analysis'))

    last_agent_report = report
    last_agent_target_column = target_column

    # report contains: analysis_results, local_recommendations, deepseek_report, deepseek_report_html
    return render_template('final_report.html', report=report, target_column=target_column)


@app.route('/final-report/pdf', methods=['GET'])
def export_final_report_pdf():
    global last_agent_report, last_agent_target_column

    if not last_agent_report:
        flash("Сначала сформируйте финальный отчёт, затем экспортируйте PDF.")
        return redirect(url_for('statistical_analysis'))

    report = last_agent_report
    target_column = last_agent_target_column

    buffer = io.BytesIO()

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            title="Final data preprocessing report",
            author="AI Agent"
        )

        story = []
        story.append(Paragraph("Final Data Preprocessing Report", styles["Title"]))
        story.append(Spacer(1, 12))

        ar = report.get("analysis_results") or {}
        recs = report.get("local_recommendations") or {}

        story.append(Paragraph("Dataset summary", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Target column: <b>{target_column or 'Not specified'}</b>", styles["BodyText"]))

        numeric_stats = ar.get("numeric_stats") or {}
        categorical_stats = ar.get("categorical_stats") or {}
        story.append(Paragraph(f"Numeric features: <b>{len(numeric_stats)}</b>", styles["BodyText"]))
        story.append(Paragraph(f"Categorical features: <b>{len(categorical_stats)}</b>", styles["BodyText"]))
        story.append(Spacer(1, 12))

        def _recs_list(items, empty_text: str):
            if not items:
                return [Paragraph(empty_text, styles["BodyText"])]
            bullets = []
            for r in items:
                feature = str(r.get("feature", "")).strip() or "N/A"
                reason = str(r.get("reason", "")).strip() or "N/A"
                bullets.append(ListItem(Paragraph(f"<b>{feature}</b> — {reason}", styles["BodyText"]), leftIndent=12))
            return [ListFlowable(bullets, bulletType="bullet", leftIndent=18)]

        story.append(Paragraph("Local heuristic recommendations", styles["Heading2"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("Drop", styles["Heading3"]))
        story.extend(_recs_list(recs.get("drop"), "No clear drop recommendations."))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Keep", styles["Heading3"]))
        story.extend(_recs_list(recs.get("keep"), "No explicit keep recommendations."))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Transform / Impute", styles["Heading3"]))
        story.extend(_recs_list(recs.get("transform"), "No transform recommendations."))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Notes", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "This PDF export is intentionally generated in English to ensure reliable rendering across environments.",
                styles["BodyText"],
            )
        )

        doc.build(story)
        buffer.seek(0)
    except Exception as e:
        flash(f"Не удалось сформировать PDF: {str(e)}")
        return redirect(url_for('statistical_analysis'))

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="final_report.pdf"
    )

@app.route("/remove-outliers", methods=["POST"])
def remove_outliers_route():
    global current_df, preprocessing_log
    
    if current_df is None:
        return jsonify({"status": "error", "message": "Нет данных для обработки"}), 400
    
    try:
        data = request.get_json()
        column = data.get("column")
        
        if column not in current_df.columns:
            return jsonify({"status": "error", "message": "Столбец не найден"}), 400
        
        # Статистика до обработки выбросов
        col_before = current_df[column].copy()
        min_before = float(col_before.min()) if not col_before.dropna().empty else None
        max_before = float(col_before.max()) if not col_before.dropna().empty else None

        # Сохраняем изменения в глобальную переменную
        current_df = remove_outliers(current_df, column)

        col_after = current_df[column]
        min_after = float(col_after.min()) if not col_after.dropna().empty else None
        max_after = float(col_after.max()) if not col_after.dropna().empty else None
        replaced_mask = (col_before != col_after) & col_before.notna() & col_after.notna()
        replaced_count = int(replaced_mask.sum())
        replaced_fraction = float(replaced_count / len(col_before.dropna())) if len(col_before.dropna()) else 0.0

        try:
            preprocessing_log.append({
                "step": "remove_outliers",
                "function": "remove_outliers",
                "column": column,
                "min_before": min_before,
                "max_before": max_before,
                "min_after": min_after,
                "max_after": max_after,
                "replaced_values_count": replaced_count,
                "replaced_values_fraction": replaced_fraction
            })
        except Exception:
            pass
        
        return jsonify({
            "status": "success",
            "message": f"Выбросы в столбце {column} успешно обработаны",
            "redirect": url_for('statistical_analysis')  # Перенаправление для обновления страницы
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Ошибка при обработке выбросов: {str(e)}"
        }), 500
#------------NEW
if __name__ == "__main__":
    app.run(debug=True)
 # Логируем для отладки

