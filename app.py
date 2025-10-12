# app.py
import os
import json
import re
import base64
import io
from datetime import timedelta, datetime
import threading

from flask import (
    Flask, render_template, redirect, url_for,
    request, flash, send_file, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Flask-WTF / WTForms
from flask_wtf import FlaskForm
from flask_wtf.csrf import generate_csrf
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, Length, Optional

# OpenAI moderation (kept - optional)
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Pillow for image format detection
from PIL import Image

# ---------------------------------------------------------------------
# Gemini (Google GenAI) integration
# ---------------------------------------------------------------------
# Install: pip install google-genai
gemini_client = None
try:
    # official import style shown in docs
    from google import genai
    from google.genai import types
    gemini_client = genai.Client()  # picks GEMINI_API_KEY from env
    print("✅ Gemini (GenAI) client initialized.")
except Exception as e:
    gemini_client = None
    print(f"⚠️ Gemini client init failed (google-genai package or env key missing): {e}")


def _detect_mime(image_bytes: bytes) -> str:
    """Return best guess mime type from image bytes using Pillow."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.upper() if img.format else None
        if fmt == "JPEG" or fmt == "JPG":
            return "image/jpeg"
        if fmt == "PNG":
            return "image/png"
        if fmt == "WEBP":
            return "image/webp"
        if fmt == "HEIC":
            return "image/heic"
        if fmt == "HEIF":
            return "image/heif"
    except Exception:
        pass
    return "image/jpeg"


def predict_room_cleanliness(image_bytes: bytes) -> dict:
    """
    Use Gemini (GenAI) Vision via google-genai to classify an image as Clean or Dirty.
    Returns a dict: {'is_clean': bool, 'score': float (0-1), 'message': str}
    """
    if gemini_client is None:
        return {"is_clean": False, "score": 0.0, "message": "Gemini client not configured."}

    try:
        mime_type = _detect_mime(image_bytes)

        # Build a strict instruction that asks Gemini to only output JSON.
        # We include the image first (as required by the GenAI image examples), then the JSON-only instruction.
        json_instruction = (
            "You are an image classifier. Given the image above, respond ONLY with a JSON object "
            "with keys: label (value 'Clean' or 'Dirty'), confidence (a number between 0.0 and 1.0), "
            "and explanation (short text). Example: "
            '{"label":"Clean","confidence":0.92,"explanation":"floor looks swept"}'
        )

        # create a Part from bytes
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        # contents: image first, then text instruction
        contents = [image_part, json_instruction]

        # call gemini model - gemini-2.5-flash is a good multimodal model; change if you have another model
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            # config can be added if needed, e.g. thinking budget or safety config
        )

        text = (response.text or "").strip()
        # Try to extract a JSON object from the returned text
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            raw_json = json_match.group(0)
            try:
                parsed = json.loads(raw_json)
                label = str(parsed.get("label", "")).strip().lower()
                confidence = float(parsed.get("confidence", 0.0))
                explanation = str(parsed.get("explanation", "")).strip()
                is_clean = (label == "clean")
                message = f"Gemini: label={label.capitalize()} conf={confidence:.2f} expl={explanation}"
                return {"is_clean": is_clean, "score": max(0.0, min(1.0, confidence)), "message": message}
            except Exception:
                # if JSON parse fails, fall through to fallback
                pass

        # Fallback: simple keyword match
        low = text.lower()
        if "clean" in low and "dirty" not in low:
            return {"is_clean": True, "score": 0.9, "message": f"Gemini (fallback): {text}"}
        elif "dirty" in low and "clean" not in low:
            return {"is_clean": False, "score": 0.9, "message": f"Gemini (fallback): {text}"}
        else:
            # ambiguous
            return {"is_clean": False, "score": 0.5, "message": f"Gemini ambiguous: {text}"}

    except Exception as e:
        return {"is_clean": False, "score": 0.0, "message": f"Gemini API error: {str(e)}"}


# ---------------------------------------------------------------------
# App config (unchanged routes / functionality retained)
# ---------------------------------------------------------------------
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret")
db_path = os.path.join(basedir, "instance", "cleaningapp.db")
os.makedirs(os.path.dirname(db_path), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", f"sqlite:///{db_path}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# Cookies
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Uploads
UPLOAD_DIR = os.path.join(basedir, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, async_mode="threading")

# Limiter: keep default in-memory for dev; configure Redis for production if you want
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)

# Make csrf_token available in templates
app.jinja_env.globals['csrf_token'] = generate_csrf

# ---------------------------------------------------------------------
# WTForms definitions
# ---------------------------------------------------------------------
class RegisterForm(FlaskForm):
    name = StringField("Username", validators=[DataRequired(), Length(min=3, max=150)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("Register")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class ReportForm(FlaskForm):
    location = StringField("Location", validators=[DataRequired(), Length(min=1, max=200)])
    notes = TextAreaField("Notes", validators=[Optional(), Length(max=2000)])
    submit = SubmitField("Record Cleanup")


class RequestForm(FlaskForm):
    location = StringField("Location", validators=[DataRequired(), Length(min=1, max=200)])
    notes = TextAreaField("Notes", validators=[Optional(), Length(max=2000)])
    submit = SubmitField("Request Cleanup")


class EditUserForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=150)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    role = StringField("Role", validators=[DataRequired(), Length(max=50)])
    submit = SubmitField("Update User")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    points = db.Column(db.Integer, default=0)
    avg_rating = db.Column(db.Float)

    # Relationships
    cleanup_requests = db.relationship(
        'CleanupReport',
        back_populates='requested_by',
        foreign_keys='CleanupReport.requested_by_id',
        cascade='all, delete-orphan'
    )

    cleanups_submitted = db.relationship(
        'CleanupReport',
        back_populates='submitted_by_cleaner',
        foreign_keys='CleanupReport.submitted_by_cleaner_id',
        cascade='all, delete-orphan'
    )

    def __repr__(self):
        return f"<User {self.username}>"



class CleanupReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200), nullable=False, index=True)
    reported_by = db.Column(db.String(150), nullable=False, index=True)
    notes = db.Column(db.Text)
    photo = db.Column(db.LargeBinary)
    photo_filename = db.Column(db.String(255))
    submitted_photo = db.Column(db.LargeBinary)
    submitted_photo_filename = db.Column(db.String(255))
    status = db.Column(db.String(50), default='Pending', index=True)
    created_at = db.Column(db.DateTime, default=db.func.now(), index=True)
    
    requested_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    requested_by = db.relationship(
        'User',
        back_populates='cleanup_requests',
        foreign_keys=[requested_by_id]
    )

    submitted_by_cleaner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    submitted_by_cleaner = db.relationship(
        'User',
        back_populates='cleanups_submitted',
        foreign_keys=[submitted_by_cleaner_id]
    )

    is_complete = db.Column(db.Boolean, default=False, index=True)
    points_awarded = db.Column(db.Integer, default=0)
    rating = db.Column(db.Integer)

    def __repr__(self):
        return f"<CleanupReport {self.id} - {self.status}>"



class CleanupRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200), nullable=False, index=True)
    notes = db.Column(db.Text)
    photo_filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    status = db.Column(db.String(20), default="pending", index=True)
    requested_by_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    requested_by = db.relationship('User', foreign_keys=[requested_by_id])

    def __repr__(self):
        return f"<CleanupRequest {self.id} {self.location} ({self.status})>"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@login_manager.user_loader
def load_user(user_id: str):
    # note: Query.get is legacy warning in SQLAlchemy 2.x; it's fine for dev
    return User.query.get(int(user_id))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_user(name: str, email: str, password: str, role: str) -> bool:
    name = name.strip()
    email = email.strip().lower()

    if User.query.filter_by(username=name).first():
        flash('Username already taken.', 'danger')
        return False
    if User.query.filter_by(email=email).first():
        flash('Email already registered.', 'danger')
        return False

    hashed = generate_password_hash(password)
    new_user = User(username=name, email=email, password=hashed, role=role)
    db.session.add(new_user)
    db.session.commit()
    flash(f'{role.capitalize()} account created!', 'success')
    return True


# (Optional) keep your OpenAI moderation function if you want additional moderation
def ai_check_image_cleanliness_with_openai(image_bytes: bytes) -> dict:
    """
    Optional: call OpenAI moderation on a base64 image. This function is left here unchanged
    so you keep that capability. It requires OPENAI_API_KEY in env if used.
    """
    try:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Example: this call may need updating per OpenAI's SDK (left as-is because this was in your original code)
        response = openai.moderations.create(model="omni-moderation-latest", input=img_b64)
        result = response["results"][0]
        flagged = result.get("flagged", False)
        categories = result.get("categories", {})
        return {"is_clean": not flagged, "message": str(categories)}
    except Exception as e:
        return {"is_clean": False, "message": f"Moderation failed: {e}"}


# ---------------------------------------------------------------------
# Socket.IO helpers
# ---------------------------------------------------------------------
CLEANERS_ROOM = "cleaners"


@socketio.on("join_cleaners")
def on_join_cleaners():
    if current_user.is_authenticated and current_user.role == 'cleaner':
        join_room(CLEANERS_ROOM)
        emit("joined", {"room": CLEANERS_ROOM})


@socketio.on("leave_cleaners")
def on_leave_cleaners():
    if current_user.is_authenticated and current_user.role == 'cleaner':
        leave_room(CLEANERS_ROOM)
        emit("left", {"room": CLEANERS_ROOM})


def notify_cleaners_new_request(req: CleanupRequest):
    payload = {
        "id": req.id,
        "location": req.location,
        "notes": (req.notes or ""),
        "photo_url": url_for('static', filename=f'uploads/{req.photo_filename}', _external=False) if req.photo_filename else None,
        "created_at": req.created_at.strftime("%Y-%m-%d %H:%M"),
        "requested_by": req.requested_by.username if req.requested_by else "Unknown"
    }

    # Slight delay ensures cleaner clients have joined the room
    threading.Timer(0.3, lambda: socketio.emit("new_cleanup_request", payload, room=CLEANERS_ROOM)).start()


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/home')
def home():
    return redirect(url_for('index'))


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/landing")
def landing():
    return render_template("landing.html")


@app.route("/layout")
def layout():
    return render_template("layout.html")


@app.route("/403")
def forbidden():
    return render_template("403.html")


@app.route('/choose_register')
def choose_register():
    return render_template("choose_register.html")


# --- Auth ---
@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if create_user(form.name.data, form.email.data, form.password.data, 'user'):
            return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/register_cleaner', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register_cleaner():
    form = RegisterForm()
    if form.validate_on_submit():
        if create_user(form.name.data, form.email.data, form.password.data, 'cleaner'):
            return redirect(url_for('login'))
    return render_template('register_cleaner.html', form=form)


@app.route('/register_admin', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register_admin():
    form = RegisterForm()
    if form.validate_on_submit():
        if create_user(form.name.data, form.email.data, form.password.data, 'admin'):
            return redirect(url_for('login'))
    return render_template('register_admin.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        password = form.password.data
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f'Welcome {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html', form=form)


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


# --- Dashboards ---
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'cleaner':
        return redirect(url_for('cleaner_dashboard'))
    else:
        return redirect(url_for('user_dashboard'))


@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    users = User.query.order_by(User.id.desc()).all()
    needs_attention = CleanupReport.query.filter(CleanupReport.status.in_(['Pending', 'Dirty - Needs Re-clean'])).order_by(CleanupReport.created_at.desc()).all()
    top_cleaners = User.query.filter_by(role='cleaner').order_by(User.points.desc()).all()
    return render_template('admin_dashboard.html', users=users, needs_attention=needs_attention, top_cleaners=top_cleaners)


@app.route('/cleaner_dashboard')
@login_required
def cleaner_dashboard():
    pending_requests = CleanupRequest.query.filter_by(status="pending").order_by(CleanupRequest.created_at.desc()).all()
    reports = CleanupReport.query.order_by(CleanupReport.created_at.desc()).all()
    top_cleaners = User.query.filter_by(role='cleaner').order_by(User.points.desc()).all()
    return render_template('cleaner_dashboard.html', pending_requests=pending_requests, reports=reports, top_cleaners=top_cleaners)


@app.route('/user_dashboard')
@login_required
def user_dashboard():
    reports = CleanupReport.query.filter_by(reported_by=current_user.username).all()
    my_requests = CleanupRequest.query.filter_by(requested_by_id=current_user.id).order_by(CleanupRequest.created_at.desc()).all()

    # Query completed cleanup reports requested by this user for rating
    completed_reports = CleanupReport.query.filter(
        CleanupReport.requested_by_id == current_user.id,
        CleanupReport.is_complete == True
    ).order_by(CleanupReport.created_at.desc()).all()

    return render_template('user_dashboard.html', reports=reports, my_requests=my_requests, completed_reports=completed_reports)


# --- Cleanup ---
@app.route('/record_cleanup', methods=['GET', 'POST'])
@login_required
def record_cleanup():
    form = ReportForm()
    if form.validate_on_submit():
        file = request.files.get('photo')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            photo_bytes = file.read()
            report = CleanupReport(
                location=form.location.data,
                reported_by=current_user.username,
                notes=form.notes.data,
                photo=photo_bytes,
                photo_filename=filename
            )
            db.session.add(report)
            current_user.points = (current_user.points or 0) + 5
            db.session.commit()
            flash("Cleanup recorded successfully! You earned 5 points.", "success")
            return redirect(url_for('user_dashboard'))
        else:
            flash("Please upload a valid image file (png/jpg/jpeg/gif).", "danger")
    return render_template('record_cleanup.html', form=form)


@app.route('/report/<int:report_id>', endpoint='view_report')
@login_required
def view_report(report_id):
    report = CleanupReport.query.get_or_404(report_id)
    return render_template('view_report.html', report=report)


@app.route('/request/<int:request_id>/submit', methods=['POST'])
@login_required
def submit_request_cleanup(request_id):
    # Get the cleanup request
    req = CleanupRequest.query.get_or_404(request_id)

    # Get uploaded file
    file = request.files.get('cleaned_image')
    if not file or not file.filename:
        flash('No file uploaded.', 'danger')
        return redirect(url_for('cleaner_dashboard'))

    if not allowed_file(file.filename):
        flash('Invalid file type.', 'danger')
        return redirect(url_for('cleaner_dashboard'))

    image_bytes = file.read()
    filename = secure_filename(file.filename)

    # Create a new CleanupReport linked to this request
    report = CleanupReport(
        location=req.location,
        reported_by=req.requested_by.username if req.requested_by else 'Unknown',
        notes=req.notes,
        photo_filename=req.photo_filename,
        submitted_photo=image_bytes,
        submitted_photo_filename=filename,
        submitted_by_cleaner=current_user,
        requested_by_id=req.requested_by.id if req.requested_by else None,
        status='Pending',
        is_complete=False
    )

    # AI evaluation
    ai_result = predict_room_cleanliness(image_bytes)
    is_clean = ai_result.get('is_clean', False)
    score = ai_result.get('score', 0.0)
    message = ai_result.get('message', '')

    if is_clean:
        report.status = 'Clean'
        report.is_complete = True
        points_awarded = 10
        report.points_awarded = points_awarded
        current_user.points = (current_user.points or 0) + points_awarded
        flash(f'Image judged Clean by AI. Points awarded! (Score: {score:.2f})', 'success')
    else:
        report.status = 'Dirty - Needs Re-clean'
        report.is_complete = False
        flash(f'Image judged Dirty by AI. Please re-clean. (Score: {score:.2f})', 'warning')

    db.session.add(report)
    db.session.commit()

    # Real-time stats update
    socketio.emit("cleanup_completed", {
        "report_id": report.id,
        "location": report.location,
        "cleaner": current_user.username
    }, room=f"user_{report.requested_by_id}")

    socketio.emit("stats_updated", {"user_id": current_user.id, "new_points": current_user.points}, room=CLEANERS_ROOM)

    return redirect(url_for('cleaner_dashboard'))





# --- Cleanup Requests ---
@app.route('/request_cleanup', methods=['GET', 'POST'])
@login_required
def request_cleanup():
    form = RequestForm()
    if form.validate_on_submit():
        location = form.location.data
        notes = form.notes.data
        file = request.files.get('photo')
        filename = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_DIR, filename))

        req = CleanupRequest(
            location=location,
            notes=notes,
            photo_filename=filename,
            requested_by=current_user
        )
        db.session.add(req)
        db.session.commit()

        notify_cleaners_new_request(req)
        flash("Cleanup request submitted successfully!", "success")
        return redirect(url_for('user_dashboard'))

    return render_template("request_cleanup.html", form=form)


# --- Reports & leaderboard ---
@app.route('/user_reports')
@login_required
def user_reports():
    reports = CleanupReport.query.filter_by(reported_by=current_user.username).all()
    return render_template('user_reports.html', reports=reports)


@app.route('/top_cleaners')
@login_required
def top_cleaners():
    cleaners = User.query.filter_by(role='cleaner').order_by(User.points.desc()).all()
    return render_template('top_cleaners.html', cleaners=cleaners)


@app.route('/user_evidence')
@login_required
def user_evidence():
    reports = CleanupReport.query.filter_by(submitted_by_cleaner=current_user).all()
    return render_template("user_evidence.html", reports=reports)



@app.route('/my_statistics')
@login_required
def my_statistics():
    user = current_user

    # Total cleanups completed by this user
    total_cleanups = CleanupReport.query.filter_by(
        submitted_by_cleaner_id=user.id,
        is_complete=True
    ).count()

    # Rewards: for now we use points, or you can create a Badge model later
    rewards_earned = user.points  # or some badge calculation

    # Calculate rank based on points (simple example)
    higher_points = User.query.filter(User.points > user.points).count()
    rank = higher_points + 1

    # Yearly goal progress (example goal of 50 cleanups)
    yearly_goal = 50
    goal_progress = int((total_cleanups / yearly_goal) * 100) if yearly_goal else 0

    # Monthly cleanup counts (for Chart.js)
    from sqlalchemy import extract, func
    from datetime import datetime
    monthly_counts = {}
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        count = (
            CleanupReport.query
            .filter_by(submitted_by_cleaner_id=user.id, is_complete=True)
            .filter(extract('year', CleanupReport.created_at) == datetime.now().year)
            .filter(extract('month', CleanupReport.created_at) == m)
            .count()
        )
        monthly_counts[months[m-1]] = count

    # Recent activity (latest 5 reports)
    recent_activities = (
        CleanupReport.query
        .filter_by(submitted_by_cleaner_id=user.id)
        .order_by(CleanupReport.created_at.desc())
        .limit(5)
        .all()
    )

    # Leaderboard – top 5 users by points
    leaderboard = (
        User.query
        .order_by(User.points.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "my_statistics.html",
        total_cleanups=total_cleanups,
        rewards_earned=rewards_earned,
        rank=rank,
        goal_progress=goal_progress,
        monthly_counts=monthly_counts,
        recent_activities=recent_activities,
        leaderboard=leaderboard
    )



@app.route('/stats')
@login_required
def stats():
    return render_template("stats.html")


# --- Admin tools ---
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if current_user.role != 'admin':
        abort(403)
    user = User.query.get_or_404(user_id)
    form = EditUserForm(obj=user)
    if form.validate_on_submit():
        user.username = form.username.data
        user.email = form.email.data
        user.role = form.role.data
        db.session.commit()
        flash("User updated successfully.", "success")
        return redirect(url_for('admin_dashboard'))
    return render_template("edit_user.html", user=user, form=form)


@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        abort(403)
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.username} deleted successfully.", "success")
    return redirect(url_for('admin_dashboard'))


@app.route('/rate_cleaner/<int:report_id>', methods=['POST'])
@login_required
def rate_cleaner(report_id):
    report = CleanupReport.query.get_or_404(report_id)

    # Authorization: only the user who requested the cleanup can rate
    if report.requested_by_id != current_user.id:
        flash("You are not authorized to rate this report.", "danger")
        return redirect(url_for('user_dashboard'))

    # Check if already rated
    if report.rating is not None:
        flash("You have already rated this report.", "warning")
        return redirect(url_for('user_dashboard'))

    # Validate rating
    try:
        rating = int(request.form.get('rating', 0))
        if rating < 1 or rating > 5:
            raise ValueError
    except (ValueError, TypeError):
        flash("Invalid rating. Please select a rating between 1 and 5.", "danger")
        return redirect(url_for('user_dashboard'))

    # Update report rating
    report.rating = rating
    db.session.commit()

    # Update cleaner's average rating
    if report.submitted_by_cleaner:
        cleaner = report.submitted_by_cleaner
        # Get all ratings for this cleaner
        ratings = db.session.query(CleanupReport.rating).filter(
            CleanupReport.submitted_by_cleaner_id == cleaner.id,
            CleanupReport.rating.isnot(None)
        ).all()
        if ratings:
            avg_rating = sum(r[0] for r in ratings) / len(ratings)
            cleaner.avg_rating = round(avg_rating, 2)
        else:
            cleaner.avg_rating = None
        db.session.commit()

    flash("Thank you for your rating!", "success")
    return redirect(url_for('user_dashboard'))


# ---------------------------------------------------------------------
# Jinja Filter
# ---------------------------------------------------------------------
@app.template_filter('b64encode')
def b64encode_filter(data):
    if data is None:
        return ''
    return base64.b64encode(data).decode('utf-8')


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
