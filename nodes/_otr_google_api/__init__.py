"""Direct Google Gemini API LLM lane for OTR.

The package is intentionally import-safe: no API key is read, no network call
fires, and no cache is written until a run explicitly selects
``google_api:slot-a`` or ``google_api:slot-b``.
"""

from .client import (
    GoogleAPIBillingOrQuotaError,
    GoogleAPIError,
    GoogleAPIKeyMissingError,
    GoogleAPIModelUnavailableError,
    GoogleAPIRequestShapeError,
    create_interaction,
    resolve_api_key,
)
from .llm import GoogleAPIBackend, make_google_api_generate_fn
from .models import (
    GOOGLE_API_BACKEND_KEY,
    GOOGLE_API_MODEL_UNSELECTED,
    GOOGLE_API_PROVIDER,
    GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT,
    GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT,
    GOOGLE_API_ROW_IDS,
    GOOGLE_API_SLOT_A_ID,
    GOOGLE_API_SLOT_B_ID,
    clear_slot_bindings,
    google_api_enabled,
    google_api_model_choices,
    google_api_run_meta,
    is_google_api_row_id,
    resolve_model_for_slot,
    set_slot_bindings,
)

__all__ = [
    "GOOGLE_API_BACKEND_KEY",
    "GOOGLE_API_MODEL_UNSELECTED",
    "GOOGLE_API_PROVIDER",
    "GOOGLE_API_RECOMMENDED_CREATIVE_DEFAULT",
    "GOOGLE_API_RECOMMENDED_TECHNICAL_DEFAULT",
    "GOOGLE_API_ROW_IDS",
    "GOOGLE_API_SLOT_A_ID",
    "GOOGLE_API_SLOT_B_ID",
    "GoogleAPIBackend",
    "GoogleAPIBillingOrQuotaError",
    "GoogleAPIError",
    "GoogleAPIKeyMissingError",
    "GoogleAPIModelUnavailableError",
    "GoogleAPIRequestShapeError",
    "clear_slot_bindings",
    "create_interaction",
    "google_api_enabled",
    "google_api_model_choices",
    "google_api_run_meta",
    "is_google_api_row_id",
    "make_google_api_generate_fn",
    "resolve_api_key",
    "resolve_model_for_slot",
    "set_slot_bindings",
]
