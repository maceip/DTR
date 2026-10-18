"""A bounded application-owned loop; inference engines never execute tools."""

import json
import time
import uuid

from .errors import RuntimeFault
from .schema import Generation, Message


def session_stream(app, session_id, content, *, turn_id=None, agent=False, **options):
    runtime = app.runtime
    session = runtime.state.session(session_id)
    if session["package"] != app.package.digest:
        raise RuntimeFault("session_package", "Session belongs to another application version")
    definition = app.package.manifest.agent if agent else None
    if agent and (definition is None or definition.model != session["model"]):
        raise RuntimeFault("agent_not_configured", "This session has no matching packaged agent")
    generation = Generation.model_validate(options.get("generation", {}) or {})
    if agent and "tools" in options:
        raise RuntimeFault("agent_tools", "Packaged agents use their declared tool list")
    turn_id = turn_id or str(uuid.uuid4())
    message = Message(role="user", content=content).model_dump(mode="json", exclude_none=True)
    durable = {key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
               for key, value in options.items() if key != "cancellation"}
    with runtime.lock:
        begun = runtime.state.begin_turn(session_id, turn_id, {
            "message": message, "options": durable, "model": session["model"], "agent": agent,
        })
        if not begun["cached"]:
            runtime.active_sessions.add(session_id)
    if begun["cached"]:
        yield {"type": "done", **begun["result"], "cached": True}
        return
    messages = list(begun["messages"])
    actions = []
    options.pop("generation", None)
    deadline = time.monotonic() + generation.timeout_seconds
    committed = False
    try:
        if definition:
            # Persist the instruction as part of canonical history exactly once.
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": definition.instruction})
            elif not str(messages[0]["content"]).startswith(definition.instruction):
                messages[0] = {**messages[0], "content": definition.instruction + "\n" + str(messages[0]["content"])}
            tools = runtime.tool_gateway.descriptions(app.package, definition.tools)
        else:
            tools = options.pop("tools", [])
        messages.append(message)
        yield {"type": "turn", "session_id": session_id, "turn_id": turn_id}
        for step in range(definition.max_steps if definition else 1):
            cancellation = options.get("cancellation")
            if cancellation and cancellation.is_set():
                raise RuntimeFault("cancelled", "Turn was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeFault("inference_timeout", "Agent turn exceeded its time budget")
            result = None
            step_generation = generation.model_copy(update={"timeout_seconds": remaining})
            for event in app.chat_stream(session["model"], messages, generation=step_generation,
                                          tools=tools, **options):
                if event["type"] == "done":
                    result = event
                else:
                    yield {**event, "step": step}
            if result is None:
                raise RuntimeFault("incomplete_generation", "Model did not complete its response")
            messages.append(result["message"])
            calls = result["message"].get("tool_calls", [])
            if not calls or not definition:
                final = {**result, "session_id": session_id, "turn_id": turn_id,
                         "cached": False, "actions": actions, "steps": step + 1}
                runtime.state.finish_turn(session_id, turn_id, messages, final)
                committed = True
                yield final
                return
            for index, call in enumerate(calls):
                # IDs come from the durable turn, never from a model's guessed ID.
                action_id = f"{turn_id}:{step}:{index}"
                yield {"type": "tool_call", "step": step, "action_id": action_id, "call": call}
                if call["name"] not in definition.tools:
                    raise RuntimeFault("tool_not_declared", "Agent requested a tool outside its declared tool list")
                action = app.invoke_tool(call["name"], call["arguments"], action_id=action_id,
                                          cancellation=cancellation)
                actions.append(action)
                messages.append({"role": "tool", "name": call["name"], "tool_call_id": call["id"],
                                 "content": json.dumps({key: action[key] for key in ("action_id", "status", "result", "error") if key in action}, ensure_ascii=False)})
                yield {"type": "tool_result", "step": step, "action": action}
                if action["status"] == "uncertain":
                    raise RuntimeFault("action_outcome_unknown", "Tool outcome requires inspection before continuing",
                                       details={"action_id": action_id})
        raise RuntimeFault("agent_step_limit", "Agent reached its configured step limit",
                           details={"max_steps": definition.max_steps, "actions": actions})
    except BaseException as exc:
        error = exc.as_dict() if isinstance(exc, RuntimeFault) else {"message": str(exc) or type(exc).__name__}
        # Keep completed tool responses on interruption. This prevents the next
        # turn from being told that an already completed action never happened.
        status = "cancelled" if isinstance(exc, (GeneratorExit, KeyboardInterrupt)) or (
            isinstance(exc, RuntimeFault) and exc.code == "cancelled") else "failed"
        if not committed:
            runtime.state.finish_turn(session_id, turn_id, messages,
                                       {"error": error, "actions": actions}, status=status)
        raise
    finally:
        with runtime.lock:
            runtime.active_sessions.discard(session_id)
