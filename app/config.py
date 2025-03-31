from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file="/home/trootech/PycharmProjects/project/.env",
        extra="ignore"

    )


# noinspection PyArgumentList
Config = Settings()
