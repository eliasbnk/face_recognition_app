from datetime import datetime


class AttendanceLogger:

    @staticmethod
    def log_attendance(attendee_name=None):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Saw {attendee_name} at {current_time}")
