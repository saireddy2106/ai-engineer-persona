from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

import os
import datetime

SCOPES = [
    "https://www.googleapis.com/auth/calendar"
]


def get_calendar_service():

    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build(
        "calendar",
        "v3",
        credentials=creds
    )


def list_events():

    service = get_calendar_service()

    now = datetime.datetime.utcnow()

    future = now + datetime.timedelta(days=7)

    events = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat() + "Z",
        timeMax=future.isoformat() + "Z",
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    return events.get("items", [])


def create_interview_event(
        title,
        start_datetime,
        end_datetime):

    service = get_calendar_service()

    event = {
        "summary": title,
        "start": {
            "dateTime": start_datetime.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_datetime.isoformat(),
            "timeZone": "Asia/Kolkata",
        },
    }

    created_event = service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    return created_event.get("htmlLink")


if __name__ == "__main__":

    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)

    start_time = tomorrow.replace(
        hour=14,
        minute=0,
        second=0,
        microsecond=0
    )

    end_time = tomorrow.replace(
        hour=15,
        minute=0,
        second=0,
        microsecond=0
    )

    link = create_interview_event(
        "AI Persona Test Interview",
        start_time,
        end_time
    )

    print("Created Successfully")
    print(link)