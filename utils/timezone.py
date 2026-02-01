"""
Timezone utilities - All timestamps in Saudi Arabia Time (AST / UTC+3)
"""
from datetime import datetime
import pytz

# Saudi Arabia timezone
SAUDI_TZ = pytz.timezone('Asia/Riyadh')

def now_saudi() -> datetime:
    """Get current time in Saudi Arabia"""
    return datetime.now(SAUDI_TZ)

def now_saudi_str() -> str:
    """Get current time in Saudi Arabia as ISO string"""
    return now_saudi().isoformat()

def now_saudi_formatted() -> str:
    """Get current time formatted for display"""
    return now_saudi().strftime("%Y-%m-%d %H:%M:%S AST")

def format_timestamp(dt: datetime = None) -> str:
    """Format a datetime for display in Saudi time"""
    if dt is None:
        dt = now_saudi()
    elif dt.tzinfo is None:
        dt = SAUDI_TZ.localize(dt)
    else:
        dt = dt.astimezone(SAUDI_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S AST")
