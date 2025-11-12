import os
import io
import json
import re
import base64
import threading
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, redirect, url_for,
    request, flash, send_file, abort, jsonify
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
try:
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    print("[OK] OpenAI client initialized (optional).")
except ImportError:
    openai = None
    print("[WARN] OpenAI package not installed (optional - app will work without it).")

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
    print("[OK] Gemini (GenAI) client initialized.")
except Exception as e:
    gemini_client = None
    print(f"[WARN] Gemini client init failed (google-genai package or env key missing): {e}")


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


def verify_cleanup_improvement(before_bytes: bytes, after_bytes: bytes) -> dict:
    """
    Use Gemini Vision to verify if the after photo shows improvement over the before photo.
    This validates user-submitted cleanup records for AI-based auto-approval.
    Returns: {'is_valid': bool, 'confidence': float (0-1), 'message': str, 'auto_approve': bool}
    """
    if gemini_client is None:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "message": "AI verification not available. Admin review required.",
            "auto_approve": False
        }

    try:
        before_mime = _detect_mime(before_bytes)
        after_mime = _detect_mime(after_bytes)

        # Create image parts
        before_part = types.Part.from_bytes(data=before_bytes, mime_type=before_mime)
        after_part = types.Part.from_bytes(data=after_bytes, mime_type=after_mime)

        # Instruction for comparison
        comparison_instruction = (
            "You are a cleanup verification AI. Compare these two images: "
            "FIRST IMAGE = Before cleanup, SECOND IMAGE = After cleanup. "
            "Analyze if the second image shows significant improvement (cleaner, less litter, organized). "
            "Respond ONLY with a JSON object with keys: "
            "is_improved (boolean: true if after is cleaner), "
            "confidence (0.0 to 1.0), "
            "details (short explanation). "
            "Example: {\"is_improved\":true,\"confidence\":0.85,\"details\":\"area is visibly cleaner with less debris\"}"
        )

        # Call Gemini with both images
        contents = [before_part, after_part, comparison_instruction]
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        text = (response.text or "").strip()
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if json_match:
            raw_json = json_match.group(0)
            try:
                parsed = json.loads(raw_json)
                is_improved = parsed.get("is_improved", False)
                confidence = float(parsed.get("confidence", 0.0))
                details = str(parsed.get("details", "")).strip()

                # Auto-approve if confidence > 0.75 and improvement detected
                auto_approve = is_improved and confidence > 0.75

                return {
                    "is_valid": is_improved,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "message": f"AI Analysis: {details}",
                    "auto_approve": auto_approve
                }
            except Exception:
                pass

        # Fallback: simple keyword match
        low = text.lower()
        if "improved" in low or "cleaner" in low:
            return {
                "is_valid": True,
                "confidence": 0.6,
                "message": "AI detected cleanup improvement (low confidence - needs admin review)",
                "auto_approve": False
            }
        else:
            return {
                "is_valid": False,
                "confidence": 0.3,
                "message": "AI could not verify cleanup improvement",
                "auto_approve": False
            }

    except Exception as e:
        app.logger.exception(f"Cleanup verification error: {e}")
        return {
            "is_valid": False,
            "confidence": 0.0,
            "message": f"AI verification error: {str(e)}",
            "auto_approve": False
        }


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

# Make datetime.now available in templates
@app.context_processor
def inject_now():
    """Make datetime.now available in all templates."""
    return {'now': datetime.now()}

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

    equipment_requests = db.relationship(
        'EquipmentRequest',
        back_populates='cleaner',
        foreign_keys='EquipmentRequest.cleaner_id',
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

    claimed_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    claimed_by = db.relationship('User', foreign_keys=[claimed_by_id])

    def __repr__(self):
        return f"<CleanupRequest {self.id} {self.location} ({self.status})>"


class EquipmentRequest(db.Model):
    """Model for cleaners to request equipment or supplies."""
    id = db.Column(db.Integer, primary_key=True)
    cleaner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    cleaner = db.relationship('User', foreign_keys=[cleaner_id])
    equipment_name = db.Column(db.String(150), nullable=False, index=True)
    quantity = db.Column(db.Integer, nullable=False)
    reason = db.Column(db.Text)
    status = db.Column(db.String(50), default='Pending', index=True)  # Pending, Approved, Rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewed_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    reviewed_by = db.relationship('User', foreign_keys=[reviewed_by_id])

    def __repr__(self):
        return f"<EquipmentRequest {self.id} {self.equipment_name} ({self.status})>"


class CleanerPenalty(db.Model):
    """Model for tracking penalties issued to cleaners by admins."""
    id = db.Column(db.Integer, primary_key=True)
    cleaner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    cleaner = db.relationship('User', foreign_keys=[cleaner_id], backref='penalties')
    reason = db.Column(db.String(255), nullable=False)
    points_deducted = db.Column(db.Integer, default=0)
    issued_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    issued_by = db.relationship('User', foreign_keys=[issued_by_id])
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<CleanerPenalty {self.id} {self.cleaner_id} -{self.points_deducted}pts>"


class StoreItem(db.Model):
    """Eco-friendly items available in the Green Store."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False, index=True)
    description = db.Column(db.Text)
    cost = db.Column(db.Integer, nullable=False)  # Cost in EcoCoins
    image_filename = db.Column(db.String(255))
    available = db.Column(db.Boolean, default=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<StoreItem {self.name} - {self.cost}ₑ₵>"


class UserCleanupRecord(db.Model):
    """User-submitted cleanup records with before/after photos."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    location = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    before_photo = db.Column(db.String(255), nullable=False)
    after_photo = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, approved, rejected
    points_awarded = db.Column(db.Integer, default=0)
    admin_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='cleanup_records')
    
    def __repr__(self):
        return f"<UserCleanupRecord {self.id} {self.location} {self.status}>"


class Transaction(db.Model):
    """Tracks all EcoCoin transactions (earn, spend, penalties)."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    item_id = db.Column(db.Integer, db.ForeignKey('store_item.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    amount = db.Column(db.Integer, nullable=False)
    transaction_type = db.Column(db.String(50), nullable=False)  # 'purchase', 'earn', 'penalty', 'bonus'
    note = db.Column(db.String(255))

    user = db.relationship('User', backref='transactions')
    item = db.relationship('StoreItem', backref='purchases')

    def __repr__(self):
        return f"<Transaction {self.id} {self.transaction_type} {self.amount}ₑ₵>"


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
@login_required
def register():
    """Public registration disabled. Only admins can create users."""
    if current_user.role != 'admin':
        flash("Registration is admin-only. Contact an administrator.", "danger")
        return redirect(url_for('dashboard'))
    return redirect(url_for('register_admin'))


@app.route('/register_cleaner', methods=['GET', 'POST'])
@login_required
def register_cleaner():
    """Public cleaner registration disabled. Only admins can create cleaners."""
    if current_user.role != 'admin':
        flash("Registration is admin-only. Contact an administrator.", "danger")
        return redirect(url_for('dashboard'))
    return redirect(url_for('register_admin'))


@app.route('/register_admin', methods=['GET', 'POST'])
@login_required
@limiter.limit("10 per minute")
def register_admin():
    """Admin-only: Create a new user with specified role."""
    if current_user.role != 'admin':
        flash("Access denied. Admins only.", "danger")
        abort(403)
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'user').lower()
        
        # Validation
        if not username or len(username) < 3:
            flash("Username must be at least 3 characters.", "warning")
            return redirect(url_for('register_admin'))
        
        if not email or '@' not in email:
            flash("Invalid email address.", "warning")
            return redirect(url_for('register_admin'))
        
        if not password or len(password) < 6:
            flash("Password must be at least 6 characters.", "warning")
            return redirect(url_for('register_admin'))
        
        if role not in ['user', 'cleaner', 'admin']:
            flash("Invalid role selected.", "warning")
            return redirect(url_for('register_admin'))
        
        # Check if user already exists
        existing = User.query.filter(
            (User.email == email) | (User.username == username)
        ).first()
        
        if existing:
            flash("Username or email already exists.", "danger")
            return redirect(url_for('register_admin'))
        
        # Create new user
        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_pw,
            role=role
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash(f"✅ {role.title()} '{username}' created successfully!", "success")
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error creating user: {e}")
            flash("Error creating user. Please try again.", "danger")
            return redirect(url_for('register_admin'))
    
    return render_template('register_admin.html')


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
    equipment_requests = EquipmentRequest.query.order_by(EquipmentRequest.created_at.desc()).all()
    
    # Get cleaners with penalty info
    cleaners = User.query.filter_by(role='cleaner').all()
    penalties_by_cleaner = {}
    for penalty in CleanerPenalty.query.all():
        penalties_by_cleaner[penalty.cleaner_id] = penalties_by_cleaner.get(penalty.cleaner_id, 0) + (penalty.points_deducted or 0)
    
    return render_template('admin_dashboard.html', users=users, needs_attention=needs_attention, top_cleaners=top_cleaners, equipment_requests=equipment_requests, cleaners=cleaners, penalties_by_cleaner=penalties_by_cleaner)


@app.route('/admin/issue_penalty/<int:cleaner_id>', methods=['POST'])
@login_required
def issue_penalty(cleaner_id):
    """Admin issues a penalty to a cleaner."""
    if current_user.role != 'admin':
        abort(403)
    
    reason = request.form.get('reason', '').strip()
    points_str = request.form.get('points_deducted', '0')
    
    if not reason:
        flash("Please provide a reason for the penalty.", "warning")
        return redirect(url_for('admin_dashboard'))
    
    try:
        points_deducted = int(points_str)
        if points_deducted < 1:
            raise ValueError("Points must be at least 1")
    except ValueError:
        flash("Points must be a positive number.", "warning")
        return redirect(url_for('admin_dashboard'))
    
    cleaner = User.query.get_or_404(cleaner_id)
    
    # Create penalty record
    penalty = CleanerPenalty(
        cleaner_id=cleaner_id,
        issued_by_id=current_user.id,
        reason=reason,
        points_deducted=points_deducted
    )
    db.session.add(penalty)
    
    # Deduct points from cleaner
    cleaner.points = max(0, (cleaner.points or 0) - points_deducted)
    
    db.session.commit()
    
    flash(f"⚠️ Penalty of {points_deducted} points issued to {cleaner.username}. Reason: {reason}", "warning")
    return redirect(url_for('admin_dashboard'))


@app.route('/cleaner_dashboard')
@login_required
def cleaner_dashboard():
    if current_user.role != 'cleaner':
        flash("Access denied: Cleaners only.", "danger")
        return redirect(url_for('dashboard'))

    # Only show unclaimed or claimed by current cleaner
    pending_requests = CleanupRequest.query.filter(
        (CleanupRequest.status == "pending") |
        ((CleanupRequest.status == "claimed") & (CleanupRequest.claimed_by_id == current_user.id))
    ).order_by(CleanupRequest.created_at.desc()).all()

    # Only show reports completed by this cleaner
    reports = CleanupReport.query.filter_by(
        submitted_by_cleaner_id=current_user.id,
        is_complete=True
    ).order_by(CleanupReport.created_at.desc()).all()

    top_cleaners = User.query.filter_by(role='cleaner').order_by(User.points.desc()).limit(5).all()

    return render_template(
        'cleaner_dashboard.html',
        pending_requests=pending_requests,
        reports=reports,
        top_cleaners=top_cleaners
    )


@app.route('/equipment_request', methods=['GET', 'POST'])
@login_required
def equipment_request():
    """Cleaner submits equipment request."""
    if current_user.role != 'cleaner':
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        equipment_name = request.form.get('equipment_name', '').strip()
        quantity_str = request.form.get('quantity', '')
        reason = request.form.get('reason', '').strip()

        if not equipment_name or not quantity_str:
            flash("Please fill out all required fields.", "warning")
            return redirect(url_for('equipment_request'))

        try:
            quantity = int(quantity_str)
            if quantity < 1:
                raise ValueError("Quantity must be at least 1")
        except ValueError:
            flash("Quantity must be a positive number.", "warning")
            return redirect(url_for('equipment_request'))

        # Create new equipment request
        eq_req = EquipmentRequest(
            cleaner_id=current_user.id,
            equipment_name=equipment_name,
            quantity=quantity,
            reason=reason,
            status='Pending'
        )
        db.session.add(eq_req)
        db.session.commit()

        flash(f"Equipment request for {equipment_name} submitted successfully!", "success")
        return redirect(url_for('cleaner_dashboard'))

    # GET: Show the form
    return render_template('equipment_request.html')


@app.route('/my_equipment_requests')
@login_required
def my_equipment_requests():
    """Show cleaner their equipment requests (pending, approved, rejected)."""
    if current_user.role != 'cleaner':
        abort(403)
    
    requests = EquipmentRequest.query.filter_by(cleaner_id=current_user.id).order_by(EquipmentRequest.created_at.desc()).all()
    return render_template('my_equipment_requests.html', requests=requests)


@app.route('/approve_equipment/<int:req_id>', methods=['POST'])
@login_required
def approve_equipment(req_id):
    """Admin approves an equipment request."""
    if current_user.role != 'admin':
        abort(403)
    
    eq_req = EquipmentRequest.query.get_or_404(req_id)
    eq_req.status = 'Approved'
    eq_req.reviewed_at = datetime.utcnow()
    eq_req.reviewed_by_id = current_user.id
    db.session.commit()
    
    flash(f"✅ Approved equipment request for {eq_req.equipment_name}.", "success")
    return redirect(url_for('admin_dashboard'))


@app.route('/reject_equipment/<int:req_id>', methods=['POST'])
@login_required
def reject_equipment(req_id):
    """Admin rejects an equipment request."""
    if current_user.role != 'admin':
        abort(403)
    
    eq_req = EquipmentRequest.query.get_or_404(req_id)
    eq_req.status = 'Rejected'
    eq_req.reviewed_at = datetime.utcnow()
    eq_req.reviewed_by_id = current_user.id
    db.session.commit()
    
    flash(f"❌ Rejected equipment request for {eq_req.equipment_name}.", "warning")
    return redirect(url_for('admin_dashboard'))


@app.route('/request/<int:request_id>/accept', methods=['POST'])
@login_required
def accept_request(request_id):
    if current_user.role != 'cleaner':
        abort(403)
    req = CleanupRequest.query.get_or_404(request_id)
    if req.status != 'pending':
        flash('This request has already been claimed.', 'warning')
        return redirect(url_for('cleaner_dashboard'))
    req.claimed_by = current_user
    req.status = 'claimed'
    db.session.commit()
    
    # Emit Socket.IO event to notify other cleaners
    socketio.emit("request_claimed", {
        "request_id": req.id,
        "location": req.location,
        "cleaner": current_user.username
    }, room=CLEANERS_ROOM)
    
    flash('Request claimed successfully!', 'success')
    return redirect(url_for('cleaner_dashboard'))


@app.route('/request/<int:request_id>/reject', methods=['POST'])
@login_required
def reject_request(request_id):
    if current_user.role != 'cleaner':
        abort(403)
    req = CleanupRequest.query.get_or_404(request_id)
    if req.status != 'claimed' or req.claimed_by_id != current_user.id:
        flash('You can only reject requests you have claimed.', 'warning')
        return redirect(url_for('cleaner_dashboard'))
    req.claimed_by = None
    req.status = 'pending'
    db.session.commit()
    
    # Emit Socket.IO event to notify other cleaners
    socketio.emit("request_reopened", {
        "request_id": req.id,
        "location": req.location,
        "cleaner": current_user.username
    }, room=CLEANERS_ROOM)
    
    flash('Request rejected and returned to pending.', 'info')
    return redirect(url_for('cleaner_dashboard'))


# =====================================================================
# GREEN STORE & ECOCOIN SYSTEM
# =====================================================================

@app.route('/green_store')
@login_required
def green_store():
    """Display available eco-friendly items in the Green Store."""
    items = StoreItem.query.filter_by(available=True).order_by(StoreItem.created_at.desc()).all()
    return render_template('green_store.html', items=items)


@app.route('/buy_item/<int:item_id>', methods=['POST'])
@login_required
def buy_item(item_id):
    """Purchase an item from the Green Store using EcoCoins."""
    item = StoreItem.query.get_or_404(item_id)
    
    if not item.available:
        flash("This item is no longer available.", "warning")
        return redirect(url_for('green_store'))
    
    if current_user.points < item.cost:
        flash(f"Not enough EcoCoins! You need {item.cost} ₑ₵ but only have {current_user.points} ₑ₵.", "danger")
        return redirect(url_for('green_store'))
    
    # Deduct points
    current_user.points -= item.cost
    
    # Create transaction record
    transaction = Transaction(
        user_id=current_user.id,
        item_id=item.id,
        amount=item.cost,
        transaction_type='purchase',
        note=f'Purchased {item.name}'
    )
    
    db.session.add(transaction)
    db.session.commit()
    
    flash(f"🎉 You bought {item.name} for {item.cost} ₑ₵!", "success")
    return redirect(url_for('green_store'))


@app.route('/my_transactions')
@login_required
def my_transactions():
    """View personal EcoCoin transaction history."""
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.timestamp.desc()).all()
    return render_template('my_transactions.html', transactions=transactions)


@app.route('/admin/store', methods=['GET', 'POST'])
@login_required
def admin_store():
    """Admin page to manage Green Store items."""
    if current_user.role != 'admin':
        abort(403)
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        cost_str = request.form.get('cost', '')
        
        if not name or not cost_str:
            flash("Name and cost are required.", "warning")
            return redirect(url_for('admin_store'))
        
        try:
            cost = int(cost_str)
            if cost < 1:
                raise ValueError("Cost must be at least 1")
        except ValueError:
            flash("Cost must be a positive number.", "warning")
            return redirect(url_for('admin_store'))
        
        filename = None
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_DIR, filename))
        
        item = StoreItem(
            name=name,
            description=description,
            cost=cost,
            image_filename=filename,
            available=True
        )
        
        db.session.add(item)
        db.session.commit()
        
        flash(f"✅ New store item '{name}' added for {cost} ₑ₵!", "success")
        return redirect(url_for('admin_store'))
    
    items = StoreItem.query.order_by(StoreItem.created_at.desc()).all()
    return render_template('admin_store.html', items=items)


@app.route('/admin/item/<int:item_id>/toggle', methods=['POST'])
@login_required
def toggle_item_availability(item_id):
    """Toggle item availability (admin only)."""
    if current_user.role != 'admin':
        abort(403)
    
    item = StoreItem.query.get_or_404(item_id)
    item.available = not item.available
    db.session.commit()
    
    status = "available" if item.available else "unavailable"
    flash(f"Item '{item.name}' is now {status}.", "info")
    return redirect(url_for('admin_store'))


@app.route('/admin/item/<int:item_id>/delete', methods=['POST'])
@login_required
def delete_store_item(item_id):
    """Delete a store item (admin only)."""
    if current_user.role != 'admin':
        abort(403)
    
    item = StoreItem.query.get_or_404(item_id)
    name = item.name
    db.session.delete(item)
    db.session.commit()
    
    flash(f"✅ Store item '{name}' has been deleted.", "info")
    return redirect(url_for('admin_store'))


@app.route('/user_dashboard')
@login_required
def user_dashboard():
    if current_user.role != "user":
        flash("Access denied", "danger")
        return redirect(url_for("index"))
    
    # Separate requests by status
    active_requests = CleanupRequest.query.filter(
        CleanupRequest.requested_by_id == current_user.id,
        CleanupRequest.status.in_(['pending', 'claimed'])
    ).order_by(CleanupRequest.created_at.desc()).all()
    
    completed_requests = CleanupRequest.query.filter(
        CleanupRequest.requested_by_id == current_user.id,
        CleanupRequest.status == 'completed'
    ).order_by(CleanupRequest.created_at.desc()).all()

    # Query completed cleanup reports for rating
    completed_reports = CleanupReport.query.filter(
        CleanupReport.requested_by_id == current_user.id,
        CleanupReport.is_complete == True
    ).order_by(CleanupReport.created_at.desc()).all()

    # Impact & Awareness data (static for now, can be updated with APIs later)
    impact_data = {
        "country": "India",
        "waste_generated": "65,000 tons/day",
        "plastic_recycled": "28%",
        "aqi": "160 (Moderate)",
        "water_pollution": "35%",
        "trend": "Slightly Improving",
        "community_impact": {
            "cleanups": 420,
            "active_cleaners": 85,
            "eco_coins": 5600,
            "waste_removed": "1.2 tons"
        },
        "tips": [
            "Use paper bags instead of plastic.",
            "Segregate waste at home.",
            "Compost kitchen waste.",
            "Avoid single-use plastics.",
            "Encourage your community to participate in cleanups."
        ]
    }

    return render_template(
        'user_dashboard.html',
        active_requests=active_requests,
        completed_requests=completed_requests,
        completed_reports=completed_reports,
        impact_data=impact_data,
        store_items=StoreItem.query.filter_by(available=True).order_by(StoreItem.created_at.desc()).all()
    )


@app.route('/impact')
@login_required
def impact():
    """Display environmental impact and awareness data."""
    # Static data for now - can be updated dynamically later with real APIs
    data = {
        "country": "India",
        "waste_generated": "65,000 tons/day",
        "plastic_recycled": "28%",
        "aqi": "160 (Moderate)",
        "water_pollution": "35%",
        "trend": "Slightly Improving",
        "community_impact": {
            "cleanups": 420,
            "active_cleaners": 85,
            "eco_coins": 5600,
            "waste_removed": "1.2 tons"
        },
        "tips": [
            "Use paper bags instead of plastic.",
            "Organize local cleanups on weekends.",
            "Compost kitchen waste to reduce landfill load.",
            "Avoid single-use plastics.",
            "Encourage your community to segregate waste.",
            "Support eco-friendly local businesses.",
            "Participate in tree-planting drives.",
            "Reduce water consumption daily."
        ]
    }
    return render_template('impact_tab.html', data=data)


# --- Cleanup ---
@app.route('/record_cleanup', methods=['GET', 'POST'])
@login_required
def record_cleanup():
    """Allow users to submit their own cleanup records with before/after photos."""
    if request.method == 'POST':
        location = request.form.get('location')
        description = request.form.get('description')
        
        before_file = request.files.get('before_photo')
        after_file = request.files.get('after_photo')
        
        # Validation
        if not location or not description:
            flash("Location and description are required.", "danger")
            return redirect(url_for('record_cleanup'))
        
        if not before_file or not before_file.filename:
            flash("Please upload a before photo.", "danger")
            return redirect(url_for('record_cleanup'))
        
        if not after_file or not after_file.filename:
            flash("Please upload an after photo.", "danger")
            return redirect(url_for('record_cleanup'))
        
        if not allowed_file(before_file.filename) or not allowed_file(after_file.filename):
            flash("Please upload valid image files (png/jpg/jpeg/gif).", "danger")
            return redirect(url_for('record_cleanup'))
        
        # Save files
        try:
            # Use datetime.now() instead of deprecated utcnow()
            timestamp = int(datetime.now().timestamp())
            before_filename = secure_filename(f"before_{timestamp}_{before_file.filename}")
            after_filename = secure_filename(f"after_{timestamp}_{after_file.filename}")
            
            before_path = os.path.join(UPLOAD_DIR, before_filename)
            after_path = os.path.join(UPLOAD_DIR, after_filename)
            
            before_file.save(before_path)
            after_file.save(after_path)
            
            # Read files for AI verification
            with open(before_path, 'rb') as f:
                before_bytes = f.read()
            with open(after_path, 'rb') as f:
                after_bytes = f.read()
            
            # Run AI verification - fully automatic, no admin approval needed
            verification_result = verify_cleanup_improvement(before_bytes, after_bytes)
            
            # Determine status and points based on AI verification
            # Check is_valid and auto_approve flags from the verification result
            if verification_result.get('auto_approve', False) and verification_result.get('is_valid', False):
                # AI confirmed cleanup improvement with high confidence
                initial_status = 'approved'
                points_earned = 25
                ai_message = f"✅ AI Verified! Cleanup improvement detected with {verification_result['confidence']*100:.0f}% confidence. +25 ₑ₵ earned!"
                auto_approved = True
            else:
                # Improvement detected but low confidence (needs manual review) or no improvement
                if verification_result.get('is_valid', False):
                    initial_status = 'pending'
                    points_earned = 0
                    ai_message = f"⏳ Cleanup improvement detected ({verification_result['confidence']*100:.0f}% confidence), but flagged for manual review."
                    auto_approved = False
                else:
                    initial_status = 'rejected'
                    points_earned = 0
                    ai_message = f"❌ AI couldn't verify cleanup improvement. {verification_result.get('message', 'Please try again with clearer photos.')}"
                    auto_approved = False
            
            # Create new cleanup record
            record = UserCleanupRecord(
                user_id=current_user.id,
                location=location,
                description=description,
                before_photo=before_filename,
                after_photo=after_filename,
                status=initial_status,
                points_awarded=points_earned
            )
            
            # Award points immediately if AI approved
            if auto_approved:
                current_user.points += points_earned
                # Log the transaction
                transaction = Transaction(
                    user_id=current_user.id,
                    amount=points_earned,
                    transaction_type='earn',
                    note=f'AI-verified cleanup at {location}'
                )
                db.session.add(transaction)
            
            db.session.add(record)
            db.session.commit()
            
            flash(f"🌱 Cleanup submitted! {ai_message}", "success")
            return redirect(url_for('user_dashboard'))
            
        except Exception as e:
            app.logger.exception("Error saving cleanup record")
            db.session.rollback()
            flash("An error occurred. Please try again.", "danger")
            return redirect(url_for('record_cleanup'))
    
    return render_template('record_cleanup.html')


@app.route('/approve_user_cleanup/<int:record_id>', methods=['POST'])
@login_required
def approve_user_cleanup(record_id):
    """Admin approves a user cleanup record and awards EcoCoins."""
    if current_user.role != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('index'))
    
    record = UserCleanupRecord.query.get_or_404(record_id)
    points_reward = 25
    
    try:
        record.status = 'approved'
        record.points_awarded = points_reward
        record.user.points = (record.user.points or 0) + points_reward
        
        # Log transaction
        transaction = Transaction(
            user_id=record.user_id,
            amount=points_reward,
            transaction_type='earn',
            note=f'Approved cleanup at {record.location}'
        )
        
        db.session.add(transaction)
        db.session.commit()
        
        flash(f"✅ Approved! {record.user.username} earned {points_reward} ₑ₵", "success")
    except Exception as e:
        app.logger.exception("Error approving cleanup")
        db.session.rollback()
        flash("An error occurred.", "danger")
    
    return redirect(request.referrer or url_for('admin_dashboard'))


@app.route('/reject_user_cleanup/<int:record_id>', methods=['POST'])
@login_required
def reject_user_cleanup(record_id):
    """Admin rejects a user cleanup record."""
    if current_user.role != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('index'))
    
    record = UserCleanupRecord.query.get_or_404(record_id)
    notes = request.form.get('notes', '')
    
    try:
        record.status = 'rejected'
        record.admin_notes = notes
        db.session.commit()
        
        flash(f"❌ Rejected {record.user.username}'s submission.", "info")
    except Exception as e:
        app.logger.exception("Error rejecting cleanup")
        db.session.rollback()
        flash("An error occurred.", "danger")
    
    return redirect(request.referrer or url_for('admin_dashboard'))


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
        submitted_by_cleaner_id=current_user.id,
        requested_by_id=req.requested_by.id if req.requested_by else None,
        status='Pending',
        is_complete=False
    )

    # --- AI verification for cleaner submission ---
    ai_result = predict_room_cleanliness(image_bytes)
    is_clean = ai_result.get('is_clean', False)
    score = ai_result.get('score', 0.0)
    message = ai_result.get('message', '')

    if not is_clean:
        flash(f"AI rejected this image. It does not appear to be a valid cleaned area. Please upload a correct photo. (Score: {score:.2f})", "danger")
        return redirect(url_for('cleaner_dashboard'))

    # If clean, continue saving report
    report.status = 'Clean'
    report.is_complete = True
    points_awarded = 10
    report.points_awarded = points_awarded
    current_user.points = (current_user.points or 0) + points_awarded

    # Mark original request as completed
    req.status = 'completed'
    req.claimed_by = current_user

    db.session.add(report)
    db.session.commit()

    flash(f"Image judged Clean by AI. Points awarded! (Score: {score:.2f})", "success")

    # Real-time update: notify all cleaners to remove it from their dashboard
    socketio.emit("request_completed", {"request_id": req.id}, room=CLEANERS_ROOM)

    # Real-time stats update
    socketio.emit("cleanup_completed", {
        "report_id": report.id,
        "location": report.location,
        "cleaner": current_user.username
    }, room=f"user_{report.requested_by_id}")

    socketio.emit("stats_updated", {"user_id": current_user.id, "new_points": current_user.points}, room=CLEANERS_ROOM)

    return redirect(url_for('cleaner_dashboard'))

def verify_cleanup_image_with_ai(image_bytes: bytes) -> dict:
    """
    Use Gemini Vision to verify if the image shows a dirty/trash/cleanup area.
    Returns: {'valid': bool, 'confidence': float (0-1), 'message': str}
    """
    if gemini_client is None:
        return {
            "valid": False,
            "confidence": 0.0,
            "message": "AI verification not available. Please try again."
        }

    try:
        mime_type = _detect_mime(image_bytes)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        # Instruction for classification
        classification_instruction = (
            "You are a cleanup area classifier. Analyze this image and determine if it shows "
            "a dirty, littered, or cluttered area that needs cleanup (trash, garbage, litter, debris, etc.). "
            "Respond ONLY with a JSON object with keys: "
            "is_cleanup_area (boolean: true if image shows area needing cleanup), "
            "confidence (0.0 to 1.0 how confident you are), "
            "reason (short explanation). "
            "Example: {\"is_cleanup_area\":true,\"confidence\":0.92,\"reason\":\"area has visible litter and debris\"}"
        )

        contents = [image_part, classification_instruction]
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        text = (response.text or "").strip()
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if json_match:
            raw_json = json_match.group(0)
            try:
                parsed = json.loads(raw_json)
                is_cleanup_area = parsed.get("is_cleanup_area", False)
                confidence = float(parsed.get("confidence", 0.0))
                reason = str(parsed.get("reason", "")).strip()

                return {
                    "valid": is_cleanup_area and confidence > 0.6,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "message": f"{reason}" if is_cleanup_area else "This doesn't appear to be a cleanup-worthy area."
                }
            except Exception:
                pass

        # Fallback: simple keyword match
        low = text.lower()
        if any(word in low for word in ['trash', 'litter', 'garbage', 'dirty', 'debris', 'messy', 'waste']):
            return {
                "valid": True,
                "confidence": 0.65,
                "message": "Image shows a cleanup-worthy area"
            }
        else:
            return {
                "valid": False,
                "confidence": 0.3,
                "message": "Could not verify if this is a cleanup area. Please try another photo."
            }

    except Exception as e:
        app.logger.exception(f"Image verification error: {e}")
        return {
            "valid": False,
            "confidence": 0.0,
            "message": f"Verification error: {str(e)}"
        }


@app.route('/verify_cleanup_image', methods=['POST'])
@login_required
def verify_cleanup_image():
    """
    Endpoint for frontend to verify if uploaded image shows a cleanup area.
    Returns JSON with verification results.
    """
    file = request.files.get('image')
    
    if not file or not file.filename:
        return jsonify({'valid': False, 'confidence': 0.0, 'message': 'No image provided'})
    
    if not allowed_file(file.filename):
        return jsonify({'valid': False, 'confidence': 0.0, 'message': 'Invalid file type. Use PNG, JPG, or GIF.'})
    
    try:
        image_bytes = file.read()
        
        # Verify with AI
        result = verify_cleanup_image_with_ai(image_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.exception("Error verifying cleanup image")
        return jsonify({'valid': False, 'confidence': 0.0, 'message': f'Error: {str(e)}'})


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
