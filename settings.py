from pydantic import BaseModel
from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel, Field, field_validator


class MySettings(BaseModel):
    tool_name: str = "Approfondisci documentazione"

@plugin
def settings_schema():
    return MySettings.schema()