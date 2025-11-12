#!/usr/bin/env python
"""Quick test to identify import/runtime errors"""

print("Testing imports...")

try:
    import os
    print("✅ os")
except Exception as e:
    print(f"❌ os: {e}")

try:
    import io
    print("✅ io")
except Exception as e:
    print(f"❌ io: {e}")

try:
    import json
    print("✅ json")
except Exception as e:
    print(f"❌ json: {e}")

try:
    import re
    print("✅ re")
except Exception as e:
    print(f"❌ re: {e}")

try:
    import base64
    print("✅ base64")
except Exception as e:
    print(f"❌ base64: {e}")

try:
    from datetime import datetime, timedelta
    print("✅ datetime, timedelta")
except Exception as e:
    print(f"❌ datetime: {e}")

try:
    from flask import Flask
    print("✅ flask")
except Exception as e:
    print(f"❌ flask: {e}")

try:
    from flask_sqlalchemy import SQLAlchemy
    print("✅ flask_sqlalchemy")
except Exception as e:
    print(f"❌ flask_sqlalchemy: {e}")

try:
    from flask_login import LoginManager
    print("✅ flask_login")
except Exception as e:
    print(f"❌ flask_login: {e}")

try:
    from flask_socketio import SocketIO
    print("✅ flask_socketio")
except Exception as e:
    print(f"❌ flask_socketio: {e}")

try:
    from werkzeug.security import generate_password_hash
    print("✅ werkzeug")
except Exception as e:
    print(f"❌ werkzeug: {e}")

try:
    from flask_migrate import Migrate
    print("✅ flask_migrate")
except Exception as e:
    print(f"❌ flask_migrate: {e}")

try:
    from flask_limiter import Limiter
    print("✅ flask_limiter")
except Exception as e:
    print(f"❌ flask_limiter: {e}")

try:
    from flask_wtf import FlaskForm
    print("✅ flask_wtf")
except Exception as e:
    print(f"❌ flask_wtf: {e}")

try:
    from PIL import Image
    print("✅ Pillow")
except Exception as e:
    print(f"❌ Pillow: {e}")

try:
    import openai
    print("✅ openai")
except Exception as e:
    print(f"❌ openai: {e}")

try:
    from google import genai
    print("✅ google-genai")
except Exception as e:
    print(f"⚠️  google-genai: {e} (optional)")

print("\n✅ All required imports are available!")
