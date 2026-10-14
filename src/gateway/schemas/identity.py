"""Request and response models of the identity domain."""

from datetime import datetime

from pydantic import BaseModel, Field


class PasswordLinkPublic(BaseModel):
    """A single-use link that sets an account's password, and whether it was mailed."""

    mail_sent: bool = Field(
        description=(
            "Whether the link was mailed to the account's address. False when this deployment cannot mail "
            "links, or the send failed."
        )
    )
    link: str | None = Field(
        default=None,
        description=(
            "The link to pass on to the account's owner yourself, set only when mail_sent is false. Choosing a "
            "password through it records the address as vouched for rather than proven, since the mailbox never "
            "saw the link."
        ),
    )
    expires_at: datetime = Field(description="When the link stops working.")


__all__ = ["PasswordLinkPublic"]
