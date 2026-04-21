import re
import uuid
from datetime import datetime
from typing import Dict, Any

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def is_valid_email(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def mock_lead_capture(name: str, email: str, platform: str) -> Dict[str, Any]:
    """Mock API to capture a lead. Validates email, normalizes casing, logs structured output."""
    if not name or not name.strip():
        return {"success": False, "error": "Missing name"}
    if not is_valid_email(email):
        return {"success": False, "error": "Invalid email"}
    if not platform or not platform.strip():
        return {"success": False, "error": "Missing platform"}

    name_clean = name.strip().title()
    email_clean = email.strip().lower()
    platform_clean = platform.strip().title()
    lead_id = f"LEAD-{uuid.uuid4().hex[:8].upper()}"

    # Required spec output
    print(f"Lead captured successfully: {name_clean}, {email_clean}, {platform_clean}")

    # Structured log
    print("=" * 60)
    print("[LEAD CAPTURE LOG]")
    print(f"  Lead ID     : {lead_id}")
    print(f"  Name        : {name_clean}")
    print(f"  Email       : {email_clean}")
    print(f"  Platform    : {platform_clean}")
    print(f"  Timestamp   : {datetime.utcnow().isoformat()}Z")
    print(f"  Status      : SUCCESS")
    print("=" * 60)

    return {
        "success": True,
        "lead_id": lead_id,
        "name": name_clean,
        "email": email_clean,
        "platform": platform_clean,
    }