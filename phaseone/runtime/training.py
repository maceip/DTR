"""Fine-tuning contract only. No dataset loading or training provider yet."""

from typing import Literal

from .errors import RuntimeFault
from .permissions import ResourceIdentity
from .schema import Contract, Name


class FineTuneRequest(Contract):
    model: Name
    dataset: ResourceIdentity
    objective: Literal["function_calling", "chat"] = "function_calling"
    method: Literal["lora"] = "lora"


def fine_tune(app, request: FineTuneRequest | dict):
    request = FineTuneRequest.model_validate(request)
    if app._model(request.model).task != "chat":
        raise RuntimeFault("wrong_operation", "This fine-tuning contract requires a chat model")
    runtime = app.runtime
    if runtime.permissions is None:
        raise RuntimeFault("permission_denied", "Fine-tuning requires explicit dataset permissions")
    access = [{"resource": request.dataset, "action": action} for action in ("read", "train")]
    try:
        with runtime.permissions.authorize(runtime.subject, access) as authorization:
            runtime.state.receipt("fine_tuning_request", {
                "package": app.package.digest, "model": request.model,
                "objective": request.objective, **authorization,
                "status": "unsupported", "dataset_read": False,
            })
            raise RuntimeFault("fine_tuning_unavailable",
                               "Fine-tuning is a contract stub; no trainer is installed. No data was read or uploaded.",
                               details={"model": request.model, "objective": request.objective,
                                        "training_started": False})
    except RuntimeFault as fault:
        if fault.code == "permission_denied":
            runtime.state.receipt("permission_denied", {
                "operation": "fine_tune", "package": app.package.digest,
                "subject": runtime.subject, "access": access,
            })
        raise
